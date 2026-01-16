"""
05a_calibrate_delays.py: Plays a MW pulse during a laser pulse, while performing time tagging throughout the sequence.
This allows measuring all the delays in the system, as well as the NV initialization duration.
If the counts are too high, the program might hang. In this case reduce the resolution or use
05b_calibrate_delays_python_histogram.py if high resolution is needed.
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
initial_delay_cycles = 500 // 4  # delay before laser (units of clock cycles = 4 ns)
laser_len_cycles = 2000 // 4  # laser duration length (units of clock cycles = 4 ns)
mw_len_cycles = 1000 // 4  # MW duration length (units of clock cycles = 4 ns)
wait_between_runs = 3000 // 4  # (4ns)
n_avg = 1e6

resolution = 12  # ns
meas_len = laser_len_cycles * 4 + 1000  # total measurement length (ns)
t_vec = np.arange(0, meas_len, 1)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": t_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as calib_delays:
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    times_st = declare_stream()  # stream for 'times'
    counts = declare(int)  # variable to save the total number of photons
    i = declare(int)  # variable used to save data
    n = declare(int)  # variable used in for loop for averaging
    n_st = declare_stream()  # stream for 'iteration'

    with for_(n, 0, n < n_avg, n + 1):
        wait(initial_delay_cycles, "AOM")  # wait before starting PL
        play("laser_ON", "AOM", duration=laser_len_cycles)

        wait(initial_delay_cycles + (laser_len_cycles - mw_len_cycles) // 2, "Yb")  # delay the microwave pulse
        play("cw", "Yb", duration=mw_len_cycles)  # play microwave pulse

        measure("readout", "SNSPD", None, time_tagging.analog(times, meas_len, counts))
        wait(wait_between_runs, "SNSPD")

        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        times_st.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len, resolution)]).save("times_hist")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

simulate = True
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=28_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, calib_delays, simulation_config)
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
    qm = qmm.open_qm(config)

    job = qm.execute(calib_delays)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["times_hist", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        times_hist, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(t_vec[::resolution] + resolution / 2, times_hist / 1000 / (resolution / u.s) / iteration)
        plt.xlabel("t [ns]")
        plt.ylabel(f"counts [kcps / {resolution}ns]")
        plt.title("Delays")
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"times_hist_data": times_hist})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
