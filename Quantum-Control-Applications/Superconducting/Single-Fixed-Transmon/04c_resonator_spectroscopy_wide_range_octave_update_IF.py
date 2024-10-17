"""
        RESONATOR SPECTROSCOPY OVER A WIDE RANGE with the Octave and updating the correction matrix in QUA
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate and LO frequencies. The Octave LO frequency is swept in the
outer loop to optimize run time.
The Octave port can be calibrated at all the involved LO frequencies and at the several intermediate frequencies.
The correction parameters are updated in QUA after having preloaded the data in QUA arrays.

The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout intermediate frequency in the configuration under "resonator_IF".

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the Octave port connected to the readout line.
    - Define the readout pulse amplitude and duration in the configuration.
    - Specify the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequencies, labeled as "resonator_IF" and "resonator_LO", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration_with_octave import *
from qualang_tools.results import progress_counter, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.octave_tools import get_correction_for_each_LO_and_IF
import matplotlib.pyplot as plt
from scipy.signal import detrend


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)
# Open the quantum machine
qm = qmm.open_qm(config)

###################
# The QUA program #
###################

n_avg = 100  # The number of averages
# The intermediate frequency sweep parameters
f_min = 21 * u.MHz
f_max = 271 * u.MHz
df = 500 * u.kHz
IFs = np.arange(f_min, f_max + 0.1, df)  # The intermediate frequency vector (+ 0.1 to add f_max to IFs)

# The LO frequency sweep parameters
f_min_lo = 4.0e9
f_max_lo = 5.0e9
df_lo = f_max - f_min
LOs = np.arange(f_min_lo, f_max_lo + 0.1, df_lo)
frequency = np.array(np.concatenate([IFs + LOs[i] for i in range(len(LOs))]))

# Get the list of intermediate frequencies at which the correction matrix will be updated in QUA and the corresponding
# correction matrix elements
corrected_IFs, c00, c01, c10, c11, offset_I, offset_Q = get_correction_for_each_LO_and_IF(
    path_to_database="",
    config=config,
    element="resonator",
    gain=0,
    LO_list=LOs,
    IF_list=IFs,
    nb_of_updates=5,
    calibrate=True,
    qm=qm,
)

with program() as resonator_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the resonator IF
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    c00_qua = declare(fixed, value=c00)  # QUA variable for c00
    c01_qua = declare(fixed, value=c01)  # QUA variable for c01
    c10_qua = declare(fixed, value=c10)  # QUA variable for c10
    c11_qua = declare(fixed, value=c11)  # QUA variable for c11
    offset_I_qua = declare(fixed, value=offset_I)
    offset_Q_qua = declare(fixed, value=offset_Q)

    with for_(i, 0, i < len(LOs) + 1, i + 1):
        set_dc_offset("resonator", "I", offset_I_qua)
        set_dc_offset("resonator", "Q", offset_Q_qua)
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, IFs)):
                # Update the frequency of the digital oscillator linked to the resonator element
                update_frequency("resonator", f)
                # Update the correction matrix only at a pre-defined set of intermediate IFs
                with switch_(f):
                    for idx, current_if in enumerate(corrected_IFs):
                        with case_(int(current_if)):
                            update_correction(
                                "resonator",
                                c00_qua[len(IFs) * i + idx],
                                c01_qua[len(IFs) * i + idx],
                                c10_qua[len(IFs) * i + idx],
                                c11_qua[len(IFs) * i + idx],
                            )
                # Measure the state of the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to deplete
                wait(depletion_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the LO iteration to get the progress bar
        save(i, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the matrix along its second dimension (of size 'n_avg') and store the results
        # (1D vector) on the OPX processor
        I_st.buffer(len(IFs)).buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
        Q_st.buffer(len(IFs)).buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
        n_st.save_all("iteration")

###############
# Run Program #
###############

# Send the QUA program to the OPX, which compiles and executes it. It will stop at the 'pause' statement.
job = qm.execute(resonator_spec)
# Creates results handles to fetch the data
res_handles = job.result_handles
# Initialize empty vectors to store the global 'I' & 'Q' results
I_tot = []
Q_tot = []
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

for i, LO in enumerate(LOs):  # Loop over the LO frequencies
    # Set the frequency of the LO source
    qm.octave.set_lo_frequency("resonator", LO)
    qm.octave.set_element_parameters_from_calibration_db("resonator", job)

    # Resume the QUA program (escape the 'pause' statement)
    job.resume()
    # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
    wait_until_job_is_paused(job)
    # Fetch the data from the last OPX run corresponding to the current LO frequency
    res_handles.get("I").wait_for_values(i + 1)
    I = res_handles.get("I").fetch_all()["value"][i]
    Q = res_handles.get("Q").fetch_all()["value"][i]
    # Update the list of global results
    I_tot.append(I)
    Q_tot.append(Q)
    # Progress bar
    progress_counter(i, len(LOs))
    # Convert results into Volts
    S = u.demod2volts(I + 1j * Q, readout_len)
    R = np.abs(S)  # Amplitude
    phase = np.unwrap(np.angle(S))  # Phase
    # Plot results
    plt.suptitle("Qubit spectroscopy")
    ax1 = plt.subplot(211)
    plt.plot((IFs + LOs[i]) / u.MHz, R, ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, sharex=ax1)
    plt.plot((IFs + LOs[i]) / u.MHz, detrend(phase), ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.pause(0.1)
    plt.tight_layout()

# Interrupt the FPGA program
job.halt()
# Convert results into Volts
I = np.concatenate(I_tot)
Q = np.concatenate(Q_tot)
S = u.demod2volts(I + 1j * Q, readout_len)
R = np.abs(S)  # Amplitude
phase = detrend(np.unwrap(np.angle(S)))  # Phase
# Final plot
plt.figure()
plt.suptitle("Qubit spectroscopy")
ax1 = plt.subplot(211)
plt.plot(frequency / u.MHz, R, ".")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
plt.subplot(212, sharex=ax1)
plt.plot(frequency / u.MHz, detrend(phase), ".")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel("Phase [rad]")
plt.pause(0.1)
plt.tight_layout()
