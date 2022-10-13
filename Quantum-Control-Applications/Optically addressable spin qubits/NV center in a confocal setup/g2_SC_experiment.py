"""
Pulsed_ODMR.py: Single NVs confocal microscopy: Intensity autocorrelation g2 using 1-channel
Author: Berk Kovos - Quantum Machines
Created: 15/08/2022
Created on QUA version: 0.1.0
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt

plt.ion()


def single_channel_g2(g2, times, counts, correlation_width: int):
    """
    Calculate the second order correlation of click times at a single counting channel

    :param g2: (QUA array of type int) - g2 measurement from the previous iteration. The size must be greater than correlation width.
    :param times: (QUA array of type int) - Click times in nanoseconds
    :param counts: (QUA int) - Number of total clicks
    :param correlation_width: (int) - Relevant correlation window to analyze data
    :return: (QUA array of type int) - Updated g2
    """
    j = declare(int)
    k = declare(int)
    difference = declare(int)
    with for_(k, 0, k < counts, k + 1):
        with for_(j, k + 1, j < counts, j + 1):
            assign(difference, times[j] - times[k])
            with if_(difference < correlation_width):
                assign(g2[difference], g2[difference] + 1)
            # If the remaining photons are outside the correlation window, go to the next click
            # This is equivalent to 'break'
            with else_():
                assign(j, counts)
    return g2


# Scan Parameters
n_avg = 1e6
correlation_width = 200 * u.ns
expected_counts = 15


with program() as g2_single_channel:
    counts = declare(int)  # variable for the number of counts on SPCM1
    times = declare(int, size=expected_counts)  # array of count clicks on SPCM1

    g2 = declare(int, value=[0 for _ in range(correlation_width)])  # array for g2 to be saved
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
        measure("long_readout", "SPCM", None, time_tagging.analog(times, long_meas_len, counts))
        g2 = single_channel_g2(g2, times, counts, correlation_width)

        with for_(p, 0, p < g2.length(), p + 1):
            save(g2[p], g2_st)
            assign(g2[p], 0)

        assign(total_counts, counts + total_counts)
        save(total_counts, total_counts_st)
        save(n, n_st)

    with stream_processing():
        g2_st.buffer(correlation_width).average().save("g2")
        total_counts_st.save("total_counts")
        n_st.save("iteration")


host = "172.16.2.103"
port = "85"
qmm = QuantumMachinesManager(host=host, port=port)

simulate = True
if simulate:
    simulation_config = SimulationConfig(duration=12000)
    job = qmm.simulate(config, g2_single_channel, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config, close_other_machines=False)
    g2_SC_job = qm.execute(g2_single_channel)
    results = fetching_tool(g2_SC_job, data_list=["total_counts", "g2", "iteration"], mode="live")
    fig = plt.figure()
    interrupt_on_close(fig, g2_SC_job)

    while results.is_processing():
        # Fetch Results
        total_cts, g2_corr, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot Data
        plt.cla()
        x_ind = np.arange(correlation_width) + 1
        x_ind = np.concatenate((x_ind[::-1] * -1, x_ind))
        g2_corr = np.concatenate((g2_corr[::-1], g2_corr))
        plt.plot(x_ind, g2_corr)
        plt.xlabel("Delay time (ns)")
        plt.ylabel("Second Order Correlation (arb. u.)")
        plt.pause(0.1)
