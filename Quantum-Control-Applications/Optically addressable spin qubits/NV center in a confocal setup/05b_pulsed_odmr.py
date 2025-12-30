"""
        Pulsed Optically Detected Magnetic Resonance (pulsed ODMR)
The program consists in playing a mw pulse and the readout pulse consecutively to measure the photon
counts received by the SPCM across varying the mw frequency.
The sequence has a reference measurement window at the end of the laser pulse to normalize the photon counts.

The data is then post-processed to determine the spin resonance frequency.
This frequency can be used to update the NV intermediate frequency in the configuration under "NV_IF_freq".

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Update the different delays in the configuration

Next steps before going to the next node:
    - Update the NV frequency, labeled as "NV_IF_freq", in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.results.data_handler import DataHandler


##################
#   Parameters   #
##################
# Parameters Definition
f_vec = np.arange(-30 * u.MHz, 70 * u.MHz, 2 * u.MHz)  # Frequency vector
n_avg = 1_000_000  # number of averages

# Determine reference readout during single laser pulse
reference_wait = initialization_len_1 // 4 - 2 * meas_len_1 // 4 - 25  # in clock cycles
reference_readout = reference_wait >= 4

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": f_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as pulsed_odmr:
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    counts_ref_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, f_vec)):
            # Update the frequency of the digital oscillator linked to the element "NV"
            update_frequency("NV", f)
            # Play the mw pulse
            play("x180" * amp(1), "NV")
            # Align for laser readout after the MW pulse
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts on stream
            # Measure reference photon counts at end of laser pulse
            if reference_readout:
                wait(reference_wait, "SPCM1")
                measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            else:
                assign(counts, 1)
            save(counts, counts_ref_st)
            wait(wait_between_runs * u.ns)

        save(n, n_st)  # save number of iterations inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_st.buffer(len(f_vec)).average().save("counts")
        counts_ref_st.buffer(len(f_vec)).average().save("counts_ref")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, pulsed_odmr, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(pulsed_odmr)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "counts_ref", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, counts_ref, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot((NV_LO_freq * 0 + f_vec) / u.MHz, counts / counts_ref, label="norm. photon counts")
        plt.xlabel("MW frequency [MHz]")
        plt.ylabel("Norm. Signal")
        plt.title("pulsed ODMR")
        plt.legend()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts_data": counts})
    save_data_dict.update({"counts_dark_data": counts_ref})
    save_data_dict.update({"normalized_data": counts / counts_ref})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
