"""
        CALIBRATE DELAYS
The program consists in playing a mw pulse during a laser pulse and while performing time tagging throughout the sequence.
This allows measuring all the delays in the system, as well as the NV initialization duration.
This version processes the data in Python, which makes it slower but works better when the counts are high.

Next steps before going to the next node:
    - Update the initial laser delay (laser_delay_1) and initialization length (initialization_len_1) in the configuration.
    - Update the delay between the laser and mw (mw_delay) and the mw length (mw_len) in the configuration.
    - Update the measurement length (meas_len_1) in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
laser_delay = 500  # delay before laser [ns]
initialization_len = 2_000  # laser duration length [ns]
mw_len = 1_000  # MW duration length [ns]
wait_between_runs = 10_000  # [ns]
n_avg = 1_000_000

buffer_len = 10  # The size of each chunk of data handled by the stream processing
meas_len = initialization_len + 2 * laser_delay  # total measurement length (ns)
t_vec = np.arange(0, meas_len, 1)

assert (initialization_len - mw_len) > 4, "The MW must be shorter than the laser pulse"

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
    n = declare(int)  # variable used in for_loop
    n_st = declare_stream()  # stream for 'iteration'

    with for_(n, 0, n < n_avg, n + 1):
        # Wait before starting the play the laser pulse
        wait(laser_delay * u.ns, "AOM1")
        # Play the laser pulse for a duration given here by "initialization_len"
        play("laser_ON", "AOM1", duration=initialization_len * u.ns)

        # Delay the microwave pulse with respect to the laser pulse so that it arrive at the middle of the laser pulse
        wait((laser_delay + (initialization_len - mw_len) // 2) * u.ns, "NV")
        # Play microwave pulse
        play("cw" * amp(1), "NV", duration=mw_len * u.ns)

        # Measure the photon counted by the SPCM
        measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len, counts))
        # Adjust the wait time between each averaging iteration
        wait(wait_between_runs * u.ns, "SPCM1")
        # Save the time tags to the stream
        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        times_st.buffer(buffer_len).save_all("times")
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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(calib_delays)
    # Get results from QUA program
    res_handles = job.result_handles
    times_handle = res_handles.get("times")
    iteration_handle = res_handles.get("iteration")
    times_handle.wait_for_values(1)
    iteration_handle.wait_for_values(1)
    # Data processing initialization
    counts_vec = np.zeros(meas_len, int)
    old_count = 0

    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    b_cont = res_handles.is_processing()
    b_last = not b_cont

    while b_cont or b_last:
        plt.cla()
        new_count = times_handle.count_so_far()
        iteration = iteration_handle.fetch_all() + 1
        # Progress bar
        progress_counter(iteration, n_avg)
        # Populate the histogram
        if new_count > old_count:
            times = times_handle.fetch(slice(old_count, new_count))["value"]
            for i in range(new_count - old_count):
                for j in range(buffer_len):
                    counts_vec[times[i][j]] += 1
            old_count = new_count
        # Plot the histogram
        plt.plot(t_vec + 0.5, counts_vec / 1000 / (1 * 1e-9) / iteration)
        plt.xlabel("t [ns]")
        plt.ylabel(f"counts [kcps / 1ns]")
        plt.title("Delays")
        plt.pause(0.1)

        b_cont = res_handles.is_processing()
        b_last = not (b_cont or b_last)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"times_handle_data": times_handle})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
