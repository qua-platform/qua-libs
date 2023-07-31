"""
cw_odmr.py: Counts photons while sweeping the frequency of the applied MW.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

f_vec = np.arange(-30 * u.MHz, 70 * u.MHz, 2 * u.MHz)  # f_max + 0.1 so that f_max is included
n_avg = 1_000_000  # number of averages
wait_between_runs = 10_000

with program() as cw_odmr:
    times = declare(int, size=100)
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, f_vec)):
            update_frequency("NV", f)  # update frequency

            align()  # align all elements

            play("cw"*amp(1), "NV", duration=long_meas_len_1 * u.ns)  # play microwave pulse
            play("laser_ON", "AOM1", duration=long_meas_len_1 * u.ns)
            measure("long_readout", "SPCM1", None, time_tagging.analog(times, long_meas_len_1, counts))

            save(counts, counts_st)  # save counts on stream

            wait(wait_between_runs * u.ns)

            align()  # align all elements

            play("cw"*amp(0), "NV", duration=long_meas_len_1 * u.ns)  # play microwave pulse
            play("laser_ON", "AOM1", duration=long_meas_len_1 * u.ns)
            measure("long_readout", "SPCM1", None, time_tagging.analog(times, long_meas_len_1, counts))

            save(counts, counts_dark_st)  # save counts on stream

            wait(wait_between_runs * u.ns)

            save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(f_vec)).average().save("counts")
        counts_dark_st.buffer(len(f_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, cw_odmr, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)

    job = qm.execute(cw_odmr)  # execute QUA program

    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "counts_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, counts_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot((NV_LO_freq * 0 + f_vec) / u.MHz, counts / 1000 / (long_meas_len_1 * 1e-9))
        plt.plot((NV_LO_freq * 0 + f_vec) / u.MHz, counts_dark / 1000 / (long_meas_len_1 * 1e-9))
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Intensity [kcps]")
        plt.title("ODMR")
        plt.pause(0.1)
    plt.show()