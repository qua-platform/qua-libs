"""
A Rabi experiment sweeping the duration of the MW pulse.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

t_vec = np.arange(4, 400, 1)  # in clock cycles
n_avg = 1_000_000

with program() as time_rabi:
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    times = declare(int, size=100)
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM1")
    wait(100, "AOM1")
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            play("x180" * amp(1), "NV", duration=t)  # pulse of varied lengths
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts
            wait(1_000 * u.ns)

            align()

            play("x180" * amp(0), "NV", duration=t)  # pulse of varied lengths
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_dark_st)  # save counts
            wait(1_000 * u.ns)

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(t_vec)).average().save("counts")
        counts_dark_st.buffer(len(t_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(config)
    # execute QUA program
    job = qm.execute(time_rabi)
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
        plt.plot(t_vec * 4, counts / 1000 / (meas_len_1 / u.s))
        plt.plot(t_vec * 4, counts_dark / 1000 / (meas_len_1 / u.s))
        plt.xlabel("Tau [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Time Rabi")
        plt.pause(0.1)
    plt.show()
