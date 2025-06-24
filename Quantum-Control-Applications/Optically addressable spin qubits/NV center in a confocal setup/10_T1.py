"""
       T1 MEASUREMENT
The program consists in measuring the photon counts (in |0> and |1> successively) received by the SPCM across
varying wait times either after initialization (start from |0>), or after a pi pulse (start from |1>).
The sequence is repeated without playing the mw pulses to measure the dark counts on the SPCM.

The data is then post-processed to determine the thermal relaxation time T1.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Having set the pi pulse amplitude and duration in the configuration

Next steps before going to the next node:
    -
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
t_vec = np.arange(4, 250, 10)  # The wait time vector in clock cycles (4ns)
n_avg = 1_000_000  # The number averaging iterations
start_from_one = False

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": t_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as T1:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    counts_dark = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)  # QUA vector for storing the time-tags
    times2 = declare(int, size=100)  # QUA vector for storing the time-tags
    times_dark = declare(int, size=100)  # QUA vector for storing the time-tags
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # T1 sequence
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            # Measure in |0>
            if start_from_one:  # Choose to start either from |0> or |1>
                play("x180" * amp(1), "NV")
            wait(t, "NV")  # variable delay before measurement
            # To measure in |0> keeping the sequence time constant
            play("x180" * amp(0), "NV")
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times1, meas_len_1, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            align()
            # Measure in |1>
            if start_from_one:  # Choose to start either from |0> or |1>
                play("x180" * amp(1), "NV")
            wait(t, "NV")  # variable delay in spin Echo
            # To measure in |1>
            play("x180" * amp(1), "NV")
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2, meas_len_1, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            # Measure dark counts
            if start_from_one:  # Choose to start either from |0> or |1>
                play("x180" * amp(0), "NV")
            wait(t, "NV")  # variable delay in spin Echo
            play("x180" * amp(0), "NV")  # Pi pulse to qubit to measure second state
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the dark counts on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times_dark, meas_len_1, counts_dark))
            save(counts_dark, counts_dark_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        counts_dark_st.buffer(len(t_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, T1, simulation_config)
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
    job = qm.execute(T1)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts1", "counts2", "counts_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, counts_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(4 * t_vec, counts1 / 1000 / (meas_len_1 / u.s), label="counts in |0>")
        plt.plot(4 * t_vec, counts2 / 1000 / (meas_len_1 / u.s), label="counts in |1>")
        plt.plot(4 * t_vec, counts_dark / 1000 / (meas_len_1 / u.s), label="dark counts")
        plt.xlabel("Wait time [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title(f"T1 - starting from  {'|1>' if start_from_one else '|0>'}")
        plt.legend()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts1_data": counts1})
    save_data_dict.update({"counts2_data": counts2})
    save_data_dict.update({"counts_dark_data": counts_dark})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
