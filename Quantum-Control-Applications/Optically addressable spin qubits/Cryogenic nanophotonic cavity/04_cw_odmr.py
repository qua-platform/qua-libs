"""
04_cw_odmr.py: Counts photons while sweeping the frequency of the applied MW.
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
f_min = 270 * u.MHz  # start of freq sweep
f_max = 280 * u.MHz  # end of freq sweep
df = 2 * u.MHz  # freq step
f_vec = np.arange(f_min, f_max + 0.1, df)  # f_max + 0.1 so that f_max is included
n_avg = 1e6  # number of averages

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": f_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as cw_odmr:
    times = declare(int, size=100)
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            # initialization
            play("laser_ON", "F_transition")
            align()
            play("laser_ON", "A_transition")
            play("switch_ON", "excited_state_mw")
            align()

            # pulse sequence
            update_frequency("Yb", f)  # update frequency
            play("cw", "Yb", duration=int(meas_len // 4))  # play microwave pulse

            align()  # align all elements

            # readout laser
            play("laser_ON", "A_transition", duration=int(meas_len // 4))
            # does it needs buffer to prevent damage?

            align()  # global align

            # decay readout
            measure("long_readout", "SNSPD", None, time_tagging.analog(times, long_meas_len, counts))

            save(counts, counts_st)  # save counts on stream
            save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(f_vec)).average().save("counts")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=5_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, cw_odmr, simulation_config)
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

    job = qm.execute(cw_odmr)  # execute QUA program

    # res_handles = job.result_handles  # get access to handles
    # counts_handle = res_handles.get("counts")
    # iteration_handle = res_handles.get("iteration")
    # counts_handle.wait_for_values(1)
    # iteration_handle.wait_for_values(1)

    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # counts = counts_handle.fetch_all()
        # iteration = iteration_handle.fetch_all()

        # Fetch results
        counts, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot((Yb_LO_freq * 0 + f_vec) / u.MHz, counts / 1000 / (long_meas_len * 1e-9))
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Intensity [kcps]")
        plt.title("ODMR")

        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts_data": counts})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
