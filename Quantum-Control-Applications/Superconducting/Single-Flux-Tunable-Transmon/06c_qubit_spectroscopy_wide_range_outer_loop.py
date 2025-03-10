"""
        QUBIT SPECTROSCOPY OVER A WIDE RANGE (OUTER LOOP)
This procedure conducts a broad 1D frequency sweep of the qubit, measuring the resonator while sweeping an
external LO source simultaneously. The external LO source is swept in the outer loop to optimize run time.
Users should update the LO source frequency using the provided API at the end of the script
(lo_source.set_freq(freqs_external[i])).

Prerequisites:
    -Identification of the resonator's resonance frequency when coupled to the qubit being studied (referred to as "resonator_spectroscopy").
    -Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    -Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    -Adjust the qubit frequency settings, labeled as "qubit_IF" and "qubit_LO", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
from qualang_tools.results import progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from time import sleep
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # The number of averages
# The intermediate frequency sweep parameters
f_min = 50 * u.MHz
f_max = 300 * u.MHz
df = 1000 * u.kHz
frequencies = np.arange(f_min, f_max + 0.1, df)  # The intermediate frequency vector (+ 0.1 to add f_max to frequencies)

# The LO frequency sweep parameters
f_min_external = 3e9 - f_min
f_max_external = 4e9 - f_max
df_external = f_max - f_min
freqs_external = np.arange(f_min_external, f_max_external + 0.1, df_external)
frequency = np.array(np.concatenate([frequencies + freqs_external[i] for i in range(len(freqs_external))]))

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": frequencies,
    "external_frequencies": freqs_external,
    "frequencies": frequency,
    "config": config,
}

###################
# The QUA program #
###################
with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(i, 0, i < len(freqs_external) + 1, i + 1):
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, frequencies)):
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
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
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
        I_st.buffer(len(frequencies)).buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
        Q_st.buffer(len(frequencies)).buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
        n_st.save_all("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)


def wait_until_job_is_paused(current_job):
    """
    Waits until the OPX FPGA reaches the pause statement.
    Used when the OPX sequence needs to be synchronized with an external parameter sweep.

    :param current_job: the job object.
    """
    while not current_job.is_paused():
        sleep(0.1)
        pass
    return True


###############
# Run Program #
###############
# Open the quantum machine
qm = qmm.open_qm(config)
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
for i in range(len(freqs_external)):  # Loop over the LO frequencies
    # Set the frequency of the LO source
    lo_source.set_freq(freqs_external[i])  # Replace by your own function and add time.sleep() if needed
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
    progress_counter(iteration, len(freqs_external))
    # Convert results into Volts
    S = u.demod2volts(I + 1j * Q, readout_len)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    # Plot results
    plt.suptitle("Qubit spectroscopy")
    ax1 = plt.subplot(211)
    plt.plot((frequencies + freqs_external[i]) / u.MHz, R, ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, sharex=ax1)
    plt.plot((frequencies + freqs_external[i]) / u.MHz, phase, ".")
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
# Save results
script_name = Path(__file__).name
data_handler = DataHandler(root_data_folder=save_dir)
save_data_dict.update({"I_data": I})
save_data_dict.update({"Q_data": Q})
save_data_dict.update({"fig_live": fig})
data_handler.additional_files = {script_name: script_name, **default_additional_files}
data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
