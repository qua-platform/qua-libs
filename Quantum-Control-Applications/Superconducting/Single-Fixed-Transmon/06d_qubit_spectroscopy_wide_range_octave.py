"""
        QUBIT SPECTROSCOPY OVER A WIDE RANGE with the Octave
This procedure conducts a broad 1D frequency sweep of the qubit, measuring the resonator while sweeping the Octave
LO frequency simultaneously. The Octave LO frequency is swept in the outer loop to optimize run time.
The Octave port can be calibrated at all the involved LO frequencies and at the median intermediate frequency.
The correction parameters are updated in Python after setting the LO frequency.

Prerequisites:
    -Identification of the resonator's resonance frequency when coupled to the qubit being studied (referred to as "resonator_spectroscopy").
    -Calibration of the Octave port connected to the qubit drive line.
    -Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    -Adjust the qubit frequency settings, labeled as "qubit_IF" and "qubit_LO", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration_with_octave import *
from qualang_tools.results import progress_counter, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt


###################
# The QUA program #
###################

n_avg = 100  # The number of averages
# The intermediate frequency sweep parameters
f_min = 1 * u.MHz
f_max = 251 * u.MHz
df = 500 * u.kHz
IFs = np.arange(f_min, f_max + 0.1, df)  # The intermediate frequency vector (+ 0.1 to add f_max to IFs)
# This is to make sure that the center IF is the one used in the config for the correction parameters to be updated.
config["elements"]["qubit"]["intermediate_frequency"] = IFs[len(IFs) // 2]

# The LO frequency sweep parameters
f_min_lo = 4.0e9
f_max_lo = 5.0e9
df_lo = f_max - f_min
LOs = np.arange(f_min_lo, f_max_lo + 0.1, df_lo)
frequency = np.array(np.concatenate([IFs + LOs[i] for i in range(len(LOs))]))

with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(i, 0, i < len(LOs) + 1, i + 1):
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, IFs)):
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", f)
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

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)


###############
# Run Program #
###############
# Open the quantum machine
qm = qmm.open_qm(config)

# Calibrate the element for each LO frequency of the sweep and the central intermediate frequency
calibrate = True
if calibrate:
    for lo in LOs:
        print(f"Calibrate (LO, IF) = ({lo/u.MHz}, {IFs[len(IFs) // 2]/u.MHz}) MHz")
        qm.calibrate_element("qubit", {lo: (IFs[len(IFs) // 2],)})

# Send the QUA program to the OPX, which compiles and executes it. It will stop at the 'pause' statement.
job = qm.execute(qubit_spec)
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
    qm.octave.set_lo_frequency("qubit", LO)
    qm.octave.set_element_parameters_from_calibration_db("qubit", job)

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
