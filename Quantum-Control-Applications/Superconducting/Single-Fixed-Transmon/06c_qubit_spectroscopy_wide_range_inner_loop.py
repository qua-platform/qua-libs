"""
        QUBIT SPECTROSCOPY OVER A WIDE RANGE (INNER LOOP)
This procedure conducts an extensive 1D frequency sweep of the qubit, measuring the resonator while sweeping an
external LO source simultaneously. In this iteration, the external LO source is swept in the inner loop to reduce noise.
Users should adjust the LO source frequency using the provided API at the end of the script
(lo_source.set_freq(freqs_external[i])).

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit in focus (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Modify the qubit frequency settings, labeled as "qubit_IF" and "qubit_LO", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
from qualang_tools.results import progress_counter, wait_until_job_is_paused
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

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i, 0, i < len(freqs_external), i + 1):
            pause()  # This waits until it is resumed from python
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
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(frequencies)).buffer(len(freqs_external)).average().save("I")
        Q_st.buffer(len(frequencies)).buffer(len(freqs_external)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)


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

# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

for i in range(n_avg):
    for j in range(len(freqs_external)):
        # Set the frequency of the LO source
        lo_source.set_freq(freqs_external[j])  # Replace by your own function and add time.sleep() if needed
        # Resume the QUA program (escape the 'pause' statement)
        job.resume()
        # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
        wait_until_job_is_paused(job)
    # Live plot
    I_handle.wait_for_values(1)
    Q_handle.wait_for_values(1)
    n_handle.wait_for_values(1)
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = n_handle.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg)
    # Convert results into Volts
    S = u.demod2volts(np.concatenate(I + 1j * Q), readout_len)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    # Plot results
    ax1 = plt.suptitle("Qubit spectroscopy")
    plt.subplot(211)
    plt.cla()
    plt.plot(frequency / u.MHz, R, ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, share=ax1)
    plt.cla()
    plt.plot(frequency / u.MHz, phase, ".")
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
