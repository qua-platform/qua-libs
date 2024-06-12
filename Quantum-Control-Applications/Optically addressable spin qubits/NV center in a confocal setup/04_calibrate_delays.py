"""
        CALIBRATE DELAYS
The program consists in playing a mw pulse during a laser pulse and while performing time tagging throughout the sequence.
This allows measuring all the delays in the system, as well as the NV initialization duration.
If the counts are too high, the program might hang. In this case reduce the resolution or use
calibrate_delays_python_histogram.py if high resolution is needed.

Next steps before going to the next node:
    - Update the initial laser delay (laser_delay_1) and initialization length (initialization_len_1) in the configuration.
    - Update the delay between the laser and mw (mw_delay) and the mw length (mw_len) in the configuration.
    - Update the measurement length (meas_len_1) in the configuration.
"""
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *


###################
# The QUA program #
###################
laser_delay = 500  # delay before laser [ns]
initialization_len = 2_000  # laser duration length [ns]
mw_len = 1_000  # MW duration length [ns]
wait_between_runs = 10_000  # [ns]
n_avg = 1_000_000

resolution = 12  # ns
# total measurement length (ns), add 2*laser_delay to ensure that the window is larger than the laser pulse
meas_len = int(initialization_len + 2 * laser_delay)
# Time vector for plotting
t_vec = np.arange(0, meas_len, 1)

assert (initialization_len - mw_len) > 4, "The MW must be shorter than the laser pulse"

with program() as calib_delays:
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    times_st = declare_stream()  # stream for 'times'
    times_st_dark = declare_stream()  # stream for 'times'
    counts = declare(int)  # variable to save the total number of photons
    i = declare(int)  # variable used to save data
    n = declare(int)  # variable used in for loop for averaging
    n_st = declare_stream()  # stream for 'iteration'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        # Wait before starting the play the laser pulse
        wait(laser_delay * u.ns, "AOM1")
        # Play the laser pulse for a duration given here by "initialization_len"
        play("laser_ON", "AOM1", duration=initialization_len * u.ns)

        # Delay the microwave pulse with respect to the laser pulse so that it arrive at the middle of the laser pulse
        wait((laser_delay + (initialization_len - mw_len) // 2) * u.ns, "NV")
        # Play microwave pulse
        play("cw" * amp(1), "NV", duration=mw_len * u.ns)

        # Measure the photon counted by the SPCM
        measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len, counts))
        # Adjust the wait time between each averaging iteration
        wait(wait_between_runs * u.ns, "SPCM1")
        # Save the time tags to the stream
        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)

        align()  # global align before measuring the dark counts

        # Wait before starting the play the laser pulse
        wait(laser_delay * u.ns, "AOM1")
        # Play the laser pulse for a duration given here by "initialization_len"
        play("laser_ON", "AOM1", duration=initialization_len * u.ns)

        # Delay the microwave pulse with respect to the laser pulse so that it arrive at the middle of the laser pulse
        wait((laser_delay + (initialization_len - mw_len) // 2) * u.ns, "NV")
        # Play microwave pulse
        play("cw" * amp(0), "NV", duration=mw_len * u.ns)

        # Measure the photon counted by the SPCM
        measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len, counts))
        # Adjust the wait time between each averaging iteration
        wait(wait_between_runs * u.ns, "SPCM1")
        # Save the time tags to the stream
        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st_dark)

        save(n, n_st)  # save number of iteration for the progress bar

    with stream_processing():
        # Directly derive the histograms in the stream processing
        times_st.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len, resolution)]).save("times_hist")
        times_st_dark.histogram([[i, i + (resolution - 1)] for i in range(0, meas_len, resolution)]).save(
            "times_hist_dark"
        )
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, calib_delays, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(calib_delays)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["times_hist", "times_hist_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        times_hist, times_hist_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(t_vec[::resolution] + resolution / 2, times_hist / 1000 / (resolution / u.s) / iteration)
        plt.xlabel("t [ns]")
        plt.ylabel(f"counts [kcps / {resolution}ns]")
        plt.title("Delays")
        plt.pause(0.1)
