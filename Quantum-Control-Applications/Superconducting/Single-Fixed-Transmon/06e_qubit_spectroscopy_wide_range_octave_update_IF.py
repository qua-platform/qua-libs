"""
        QUBIT SPECTROSCOPY OVER A WIDE RANGE with the Octave and updating the correction matrix in QUA
This procedure conducts a broad 1D frequency sweep of the qubit, measuring the resonator while sweeping the Octave
LO frequency simultaneously. The Octave LO frequency is swept in the outer loop to optimize run time.
The Octave port can be calibrated at all the involved LO frequencies and at the several intermediate frequencies.
The correction parameters are updated in QUA after having preloaded the data in QUA arrays.

Prerequisites:
    -Identification of the resonator's resonance frequency when coupled to the qubit being studied (referred to as "resonator_spectroscopy").
    -Calibration of the Octave port connected to the qubit drive line.
    -Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    -Adjust the qubit frequency settings, labeled as "qubit_IF" and "qubit_LO", in the configuration.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from qualang_tools.results import progress_counter, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.octave_tools import get_correction_for_each_LO_and_IF
import matplotlib.pyplot as plt


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
f_min = 1 * u.MHz
f_max = 251 * u.MHz
df = 2000 * u.kHz
IFs = np.arange(f_min, f_max + 0.1, df)

# The LO frequency sweep parameters
f_min_external = 5.001e9 - f_min
f_max_external = 5.5e9 - f_max
df_external = f_max - f_min
LOs = np.arange(f_min_external, f_max_external + df_external / 2, df_external)
# The total frequency vector (LOs + IFs)
frequency = np.array(np.concatenate([IFs + LOs[i] for i in range(len(LOs))]))

# Get the list of intermediate IFs at which the correction matrix will be updated in QUA and the corresponding
# correction matrix elements
corrected_IFs, c00, c01, c10, c11, offset_I, offset_Q = get_correction_for_each_LO_and_IF(
    path_to_database="",
    config=config,
    element="qubit",
    gain=0,
    LO_list=LOs,
    IF_list=IFs,
    nb_of_updates=5,
    calibrate=True,
    qm=qm,
)

with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the qubit frequency
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
        set_dc_offset("qubit", "I", offset_I_qua)
        set_dc_offset("qubit", "Q", offset_Q_qua)
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, IFs)):
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", f)
                # Update the correction matrix only at a pre-defined set of intermediate IFs
                with switch_(f):
                    for idx, current_if in enumerate(corrected_IFs):
                        with case_(int(current_if)):
                            update_correction(
                                "qubit",
                                c00_qua[len(IFs) * i + idx],
                                c01_qua[len(IFs) * i + idx],
                                c10_qua[len(IFs) * i + idx],
                                c11_qua[len(IFs) * i + idx],
                            )
                # Play the saturation pulse to put the qubit in a mixed state
                play("saturation", "qubit")
                # Align the two elements to measure after playing the qubit pulse.
                # One can also measure the resonator while driving the qubit (2-tone spectroscopy) by commenting the 'align'
                align("qubit", "resonator")
                # Measure the state of the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
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
job = qm.execute(qubit_spec)
# Creates results handles to fetch the data
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
n_handle = res_handles.get("iteration")
# Initialize empty vectors to store the global 'I' & 'Q' results
I_tot = []
Q_tot = []
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
for i, LO in enumerate(LOs):  # Loop over the LO IFs
    # Set the frequency and gain of the LO source
    qm.octave.set_lo_frequency("qubit", LO)
    qm.octave.set_rf_output_gain("qubit", 0)
    # Update the correction parameters
    # qm.octave.set_element_parameters_from_calibration_db("qubit", job)

    # Resume the QUA program (escape the 'pause' statement)
    job.resume()
    # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
    wait_until_job_is_paused(job)
    # Wait until the data of this run is processed by the stream processing
    I_handle.wait_for_values(i + 1)
    Q_handle.wait_for_values(i + 1)
    n_handle.wait_for_values(i + 1)
    # Fetch the data from the last OPX run corresponding to the current LO frequency
    I = np.concatenate(I_handle.fetch(i)["value"])
    Q = np.concatenate(Q_handle.fetch(i)["value"])
    iteration = n_handle.fetch(i)["value"][0]
    # Update the list of global results
    I_tot.append(I)
    Q_tot.append(Q)
    # Progress bar
    progress_counter(iteration, len(LOs))
    # Convert results into Volts
    S = u.demod2volts(I + 1j * Q, readout_len)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    # Plot results
    plt.suptitle("Qubit spectroscopy")
    ax1 = plt.subplot(211)
    plt.plot((IFs + LOs[i]) / u.MHz, R, ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, sharex=ax1)
    plt.plot((IFs + LOs[i]) / u.MHz, phase, ".")
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
phase = np.angle(S)  # Phase
# Final plot
plt.figure()
plt.suptitle("Qubit spectroscopy")
ax1 = plt.subplot(211)
plt.plot(frequency / u.MHz, R, ".")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
plt.subplot(212, sharex=ax1)
plt.plot(frequency / u.MHz, phase, ".")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel("Phase [rad]")
plt.pause(0.1)
plt.tight_layout()
