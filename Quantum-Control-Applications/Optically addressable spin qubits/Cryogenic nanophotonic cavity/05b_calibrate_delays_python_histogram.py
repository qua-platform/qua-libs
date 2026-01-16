"""
Plays a MW pulse during a laser pulse, while performing time tagging throughout the sequence. This allows measuring all
the delays in the system, as well as the NV initialization duration.
This version process the data in Python, which makes it slower but works better when the counts are high.
"""

from qm import QuantumMachinesManager
from qm.qua import *
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

buffer_len = 10
meas_len = (laser_len_cycles + 2 * initial_delay_cycles) * 4  # total measurement length (ns)
t_vec = np.arange(0, meas_len, 1)

assert (laser_len_cycles - mw_len_cycles) > 4, "The MW must be shorter than the laser pulse"

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
        times_st.buffer(buffer_len).save_all("times")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

qm = qmm.open_qm(config)

job = qm.execute(calib_delays)

res_handles = job.result_handles
times_handle = res_handles.get("times")
iteration_handle = res_handles.get("iteration")
times_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)
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

    if new_count > old_count:
        times = times_handle.fetch(slice(old_count, new_count))["value"]
        for i in range(new_count - old_count):
            for j in range(buffer_len):
                counts_vec[times[i][j]] += 1
        old_count = new_count

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
