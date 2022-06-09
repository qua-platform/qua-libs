"""
Measures T2.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

t_min = 4  # in clock cycles units (must be >= 4)
t_max = 500  # in clock cycles units
dt = 20  # in clock cycles units
t_vec = np.arange(t_min, t_max + 0.1, dt)  # +0.1 to include t_max in array
n_avg = 1e6

with program() as hahn_echo:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)
    times2 = declare(int, size=100)
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM")
    wait(100)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(t, t_min, t <= t_max, t + dt):
            play("pi_half", "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("pi", "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("pi_half", "NV")  # Pi/2 pulse to qubit

            align()

            play("laser_ON", "AOM")
            measure("readout", "SPCM", None, time_tagging.analog(times1, meas_len, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(100, "AOM")

            align()

            play("pi_half", "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("pi", "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            frame_rotation_2pi(0.5, "NV")  # Turns next pulse to -x
            play("pi_half", "NV")  # Pi/2 pulse to qubit
            reset_frame("NV")

            align()

            play("laser_ON", "AOM")
            measure("readout", "SPCM", None, time_tagging.analog(times2, meas_len, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(100, "AOM")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

qm = qmm.open_qm(config)

job = qm.execute(hahn_echo)  # execute QUA program

res_handles = job.result_handles  # get access to handles
counts1_handle = res_handles.get("counts1")
counts2_handle = res_handles.get("counts2")
iteration_handle = res_handles.get("iteration")
counts1_handle.wait_for_values(1)
counts2_handle.wait_for_values(1)
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

ax1 = f.add_subplot(111)
while b_cont or b_last:
    ax1.cla()
    counts1 = counts1_handle.fetch_all()
    counts2 = counts2_handle.fetch_all()
    iteration = iteration_handle.fetch_all() + 1
    if iteration / n_avg > next_percent:
        percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
        print(f"{percent}%", end=" ")
        next_percent = percent / 100 + 0.1  # Print every 10%

    ax1.plot(4 * t_vec, counts1 / 1000 / (meas_len * 1e-9), counts2 / 1000 / (meas_len * 1e-9))
    ax1.xlabel("Tau [ns]")
    ax1.ylabel("Intensity [kcps]")
    ax1.title("Hahn Echo")
    ax2 = ax1.twiny()
    ax2.set_xticks(2 * 4 * t_vec)
    ax2.xlabel("Total Time [ns]")
    plt.pause(0.1)

    b_cont = res_handles.is_processing()
    b_last = not (b_cont or b_last)

print("")
