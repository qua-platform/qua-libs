"""
cw_odmr.py: Counts photons while sweeping the frequency of the applied MW.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

f_min = -30 * u.MHz  # start of freq sweep
f_max = 70 * u.MHz  # end of freq sweep
df = 2 * u.MHz  # freq step
f_vec = np.arange(f_min, f_max + 0.1, df)  # f_max + 0.1 so that f_max is included
n_avg = 1e6  # number of averages

with program() as cw_odmr:
    times = declare(int, size=100)
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            update_frequency("NV", f)  # update frequency
            align()  # align all elements

            play("cw", "NV", duration=int(long_meas_len // 4))  # play microwave pulse
            play("laser_ON", "AOM", duration=int(long_meas_len // 4))
            measure("long_readout", "SPCM", None, time_tagging.analog(times, long_meas_len, counts))

            save(counts, counts_st)  # save counts on stream
            save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(f_vec)).average().save("counts")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

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
fig = plt.figure(figsize=(8, 11))
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

b_cont = results.is_processing()
b_last = not b_cont

ax1 = fig.add_subplot(111)
while b_cont or b_last:

    # counts = counts_handle.fetch_all()
    # iteration = iteration_handle.fetch_all()

    # Fetch results
    counts, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg)
    # Plot data
    ax1.cla()
    ax1.plot(NV_LO_freq + f_vec, counts / 1000 / (long_meas_len * 1e-9))
    ax1.xlabel("Frequency [Hz]")
    ax1.ylabel("Intensity [kcps]")
    ax1.title("ODMR")
    ax2 = ax1.twiny()
    ax2.set_xticks(f_vec)
    ax2.xlabel("Intermediate Frequency [Hz]")
    plt.pause(0.1)

    b_cont = results.is_processing()
    b_last = not (b_cont or b_last)
