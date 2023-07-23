"""
calibrate_delays.py: Plays a MW pulse during a laser pulse, while performing time tagging throughout the sequence.
This allows measuring all the delays in the system, as well as the NV initialization duration.
If the counts are too high, the program might hang. In this case reduce the resolution or use
calibrate_delays_python_histogram.py if high resolution is needed.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

initial_delay = 500  # delay before laser [ns]
laser_len = 2_000  # laser duration length [ns]
mw_len = 1_000  # MW duration length [ns]
wait_between_runs = 10_000  # [ns]
n_avg = 1_000_000 

resolution = 12  # ns
meas_len = int(laser_len + 1000) # total measurement length (ns)
t_vec = np.arange(0, meas_len, 1)

with program() as calib_delays:
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    times_st = declare_stream()  # stream for 'times'
    counts = declare(int)  # variable to save the total number of photons
    i = declare(int)  # variable used to save data
    n = declare(int)  # variable used in for loop for averaging
    n_st = declare_stream()  # stream for 'iteration'

    with for_(n, 0, n < n_avg, n + 1):

        wait(initial_delay * u.ns, "AOM1")  # wait before starting PL
        play("laser_ON", "AOM1", duration=laser_len * u.ns)

        wait((initial_delay + (laser_len - mw_len) // 2) * u.ns, "NV")  # delay the microwave pulse
        play("cw", "NV", duration=mw_len * u.ns)  # play microwave pulse

        measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len, counts))
        wait(wait_between_runs * u.ns, "SPCM1")

        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream
        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        times_st.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len, resolution)]).save("times_hist")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, calib_delays, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)

    job = qm.execute(calib_delays)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration[0], n_avg, start_time=results.get_start_time())
    results = fetching_tool(job, data_list=["times_hist", "iteration"])
    times_hist, iteration = results.fetch_all()
    # Plot data
    # plt.cla()
    plt.plot(t_vec[::resolution] + resolution / 2, times_hist / 1000 / (resolution / u.s) / iteration)
    plt.xlabel("t [ns]")
    plt.ylabel(f"counts [kcps / {resolution}ns]")
    plt.title("Delays")
    # plt.pause(0.1)
    plt.show()
