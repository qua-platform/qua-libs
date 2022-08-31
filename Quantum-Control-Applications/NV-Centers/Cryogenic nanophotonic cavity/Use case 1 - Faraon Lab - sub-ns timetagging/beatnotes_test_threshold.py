from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
import datetime

###################
# The QUA program #
###################

n_avg = 6e6 * 6

resolution = 50  # ps
t_vec = np.arange(0, meas_len * 1e3, 1)

ttl_wait = 500 // 4

AC_len = 250  # 1 us


with program() as calib_delays:
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    times_st = declare_stream()  # stream for 'times'
    counts = declare(int)  # variable to save the total number of photons
    i = declare(int)  # variable used to save data
    n = declare(int)  # variable used in for loop for averaging
    n_st = declare_stream()  # stream for 'iteration'

    with for_(n, 0, n < n_avg, n + 1):

        reset_phase("A_transition")
        play("laser_ON", "A_transition", duration=AC_len)

        wait(ttl_wait, "AOM")  # AOM is dummy channel for the TTL switch

        align("AOM", "SNSPD")

        wait(258, "SNSPD")  # 258 cycles is the time it takes for the photon signal to arrive to the OPX input

        play("laser_ON", "AOM", duration=40 // 4)
        measure("readout", "SNSPD", None, time_tagging.high_res(times, meas_len, counts))

        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        times_st.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len * int(1e3), resolution)]).save(
            "times_hist"
        )
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)
# Update signal threshold before opening the quantum machine
new_signal_threshold = 600
config["elements"]["SNSPD"]["outputPulseParameters"]["signalThreshold"] = new_signal_threshold
# Open a quantum machine
qm = qmm.open_qm(config)
# Execute the QUA program
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
    plt.xlabel("t [ps]")
    plt.ylabel(f"counts [kcps / {resolution}ps]")
    plt.pause(0.1)

# Save results
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
np.savez(
    timestamp + "_high_res_beatnote", t_vec, times_hist, resolution, iteration, new_signal_threshold, ttl_wait, AC_len
)
