from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt


def MZI_g2(g2, times_1, counts_1, times_2, counts_2, correlation_width: int):
    """
    Calculate the second order correlation of click times between two counting channels

    :param g2: (QUA array of type int) - g2 measurement from the previous iteration. The size must be greater than correlation width.
    :param times_1: (QUA array of type int) - Click times in nanoseconds from channel 1
    :param counts_1: (QUA int) - Number of total clicks at channel 1
    :param times_2: (QUA array of type int) - Click times in nanoseconds from channel 2
    :param counts_2: (QUA int) - Number of total clicks at channel 2
    :param correlation_width: (int) - Relevant correlation window to analyze data
    :return: (QUA array of type int) - Updated g2
    """
    j = declare(int)
    k = declare(int)
    diff = declare(int)
    diff_ind = declare(int)
    lower_index_tracker = declare(int)
    # set the lower index tracker for each dataset
    assign(lower_index_tracker, 0)
    with for_(k, 0, k <= counts_1, k + 1):
        with for_(j, lower_index_tracker, j <= counts_2, j + 1):
            assign(diff, times_2[j] - times_1[k])
            # if correlation is outside the relevant window move to the next photon
            with if_(diff > correlation_width):
                assign(j, counts_2 + 1)
            with elif_((diff < correlation_width) & (diff > -correlation_width)):
                assign(diff_ind, diff + correlation_width)
                assign(g2[diff_ind], g2[diff_ind] + 1)
            # Track and evolve the lower bound forward every time a photon falls behind the lower bound
            with elif_(diff < -correlation_width):
                assign(lower_index_tracker, lower_index_tracker + 1)
    return g2


# Scan Parameters
n_avg = 1e4
correlation_width = 200 * u.ns
expected_counts = 15

with program() as g2_two_channel:
    counts_1 = declare(int)  # variable for the number of counts on SPCM1
    counts_2 = declare(int)  # variable for the number of counts on SPCM2
    times_1 = declare(int, size=expected_counts)  # array of count clicks on SPCM1
    times_2 = declare(int, size=expected_counts)  # array of count clicks on SPCM2

    g2 = declare(int, value=[0 for _ in range(2 * correlation_width)])  # array for g2 to be saved
    total_counts = declare(int)

    # Streamables
    g2_st = declare_stream()  # g2 stream
    total_counts_st = declare_stream()  # total counts stream
    n_st = declare_stream()  # iteration stream

    # Variables for computation
    p = declare(int)  # Some index to run over
    n = declare(int)  # n: repeat index

    with for_(n, 0, n < n_avg, n + 1):
        play("laser_ON", "AOM", duration=long_meas_len // 4)
        measure("long_readout", "SPCM", None, time_tagging.analog(times_1, long_meas_len, counts_1))
        measure("long_readout", "SPCM2", None, time_tagging.analog(times_2, long_meas_len, counts_2))
        g2 = MZI_g2(g2, times_1, counts_1, times_2, counts_2, correlation_width)

        with for_(p, 0, p < g2.length(), p + 1):
            save(g2[p], g2_st)
            assign(g2[p], 0)

        assign(total_counts, counts_1 + counts_2 + total_counts)
        save(total_counts, total_counts_st)
        save(n, n_st)

    with stream_processing():
        g2_st.buffer(correlation_width * 2).average().save("g2")
        total_counts_st.save("total_counts")
        n_st.save("iteration")

host = '172.16.2.103'
port = '85'
qmm = QuantumMachinesManager(host=host, port=port)

simulate = False
if simulate:
    simulation_config = SimulationConfig(duration=12000)
    job = qmm.simulate(config, g2_two_channel, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config, close_other_machines=False)
    job = qm.execute(g2_two_channel)
    results = fetching_tool(job, data_list=["total_counts", "g2", "iteration"], mode="live")
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch Results
        total_cts, g2_corr, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot Data
        plt.cla()
        x_ind = np.arange(correlation_width) + 1
        x_ind = np.concatenate((x_ind[::-1] * -1, x_ind))
        # sum up the inverse counts as the two detectors are interchangeable
        g2_final = g2_corr + g2_corr[::-1]
        plt.plot(x_ind, g2_final)
        plt.xlabel("Delay time (ns)")
        plt.ylabel("Second Order Correlation (arb. u.)")
        plt.pause(0.1)
