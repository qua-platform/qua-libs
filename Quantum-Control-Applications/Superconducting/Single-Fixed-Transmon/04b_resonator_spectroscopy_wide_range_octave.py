"""
        RESONATOR SPECTROSCOPY OVER A WIDE RANGE with the Octave
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate and LO frequencies. The Octave LO frequency is swept in the
outer loop to optimize run time.
The Octave port can be calibrated at all the involved LO frequencies and at the median intermediate frequency.
The correction parameters are updated in Python after setting the LO frequency.
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
import matplotlib.pyplot as plt
from scipy.signal import detrend
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # The number of averages
# The intermediate frequency sweep parameters
f_min = 21 * u.MHz
f_max = 271 * u.MHz
df = 500 * u.kHz
IFs = np.arange(f_min, f_max + 0.1, df)  # The intermediate frequency vector (+ 0.1 to add f_max to IFs)
# This is to make sure that the center IF is the one used in the config for the correction parameters to be updated.
config["elements"]["resonator"]["intermediate_frequency"] = IFs[len(IFs) // 2]

# The LO frequency sweep parameters
f_min_lo = 4.0e9
f_max_lo = 5.0e9
df_lo = f_max - f_min
LOs = np.arange(f_min_lo, f_max_lo + 0.1, df_lo)
frequency = np.array(np.concatenate([IFs + LOs[i] for i in range(len(LOs))]))

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": IFs,
    "LO_frequencies": LOs,
    "frequencies": frequency,
    "config": config,
}

###################
# The QUA program #
###################
with program() as resonator_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the resonator IF
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(i, 0, i < len(LOs) + 1, i + 1):
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, IFs)):
                # Update the frequency of the digital oscillator linked to the resonator element
                update_frequency("resonator", f)
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
        print(f"Calibrate (LO, IF) = ({lo / u.MHz}, {IFs[len(IFs) // 2] / u.MHz}) MHz")
        qm.calibrate_element("resonator", {lo: (IFs[len(IFs) // 2],)})

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
# Save results
script_name = Path(__file__).name
data_handler = DataHandler(root_data_folder=save_dir)
save_data_dict.update({"I_data": I})
save_data_dict.update({"Q_data": Q})
save_data_dict.update({"fig_live": fig})
data_handler.additional_files = {script_name: script_name, **default_additional_files}
data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
