"""
Plays a MW pulse during a laser pulse, while performing time tagging throughout the sequence. This allows measuring all
the delays in the system, as well as the NV initialization duration.
If the counts are too high, the program might hang. In this case reduce the resolution or use
calibrate_delays_python_histogram.py if high resolution is needed.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

initial_delay_cycles = 500 // 4  # delay before laser (units of clock cycles = 4 ns)
laser_len_cycles = 2000 // 4  # laser duration length (units of clock cycles = 4 ns)
mw_len_cycles = 1000 // 4  # MW duration length (units of clock cycles = 4 ns)
wait_between_runs = 3000 // 4  # (4ns)
n_avg = 1e6

resolution = 12  # ns
meas_len = laser_len_cycles * 4 + 1000  # total measurement length (ns)
t_vec = np.arange(0, meas_len, 1)

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

        wait(initial_delay_cycles + (laser_len_cycles - mw_len_cycles) // 2, "NV")  # delay the microwave pulse
        play("const", "NV", duration=mw_len_cycles)  # play microwave pulse

        measure("readout", "SPCM", None, time_tagging.analog(times, meas_len, counts))
        wait(wait_between_runs, "SPCM")

        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        times_st.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len, resolution)]).save("times_hist")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(calib_delays)

res_handles = job.result_handles
times_hist_handle = res_handles.get("times_hist")
iteration_handle = res_handles.get("iteration")
times_hist_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)


def on_close(event):
    event.canvas.stop_event_loop()
    job.halt()


f = plt.figure()
f.canvas.mpl_connect("close_event", on_close)
next_percent = 0.1  # First time print 10%
print("Progress =", end=" ")

b_cont = res_handles.is_processing()
b_last = not b_cont

while b_cont or b_last:
    plt.cla()
    times_hist = times_hist_handle.fetch_all()
    iteration = iteration_handle.fetch_all() + 1
    if iteration / n_avg > next_percent:
        percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
        print(f"{percent}%", end=" ")
        next_percent = percent / 100 + 0.1  # Print every 10%

    plt.plot(t_vec[::resolution] + resolution / 2, times_hist / 1000 / (resolution * 1e-9) / iteration)
    plt.xlabel("t [ns]")
    plt.ylabel(f"counts [kcps / {resolution}ns]")
    plt.title("Delays")
    plt.pause(0.1)

    b_cont = res_handles.is_processing()
    b_last = not (b_cont or b_last)

print("")
