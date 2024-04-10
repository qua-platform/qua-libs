"""
       HAHN ECHO MEASUREMENT (T2)
The program consists in playing two Hahn echo sequences successively (first ending with x90 and then with -x90)
and measure the photon counts received by the SPCM across varying idle times.
The sequence is repeated without playing the mw pulses to measure the dark counts on the SPCM.

The data is then post-processed to determine the coherence time T2.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Having set the pi pulse amplitude and duration in the configuration

Next steps before going to the next node:
    -
"""
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array


###################
# The QUA program #
###################

# The time vector for varying the idle time in clock cycles (4ns)
t_vec = np.arange(4, 500, 20)
n_avg = 1_000_000

with program() as hahn_echo:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    counts_dark = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)  # QUA vector for storing the time-tags
    times2 = declare(int, size=100)  # QUA vector for storing the time-tags
    times_dark = declare(int, size=100)  # QUA vector for storing the time-tags
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # Hahn echo sequence
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            # First Ramsey sequence with x90 - idle time - x90
            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # Variable idle time
            play("x180" * amp(1), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # Variable idle time
            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            align()  # Play the laser pulse after the Echo sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times1, meas_len_1, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            align()
            # Second Ramsey sequence with x90 - idle time - -x90
            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # Variable idle time
            play("x180" * amp(1), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("-x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            align()  # Play the laser pulse after the Echo sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2, meas_len_1, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            align()
            # Third Ramsey sequence for measuring the dark counts
            play("x90" * amp(0), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # Variable idle time
            play("x180" * amp(0), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("-x90" * amp(0), "NV")  # Pi/2 pulse to qubit
            align()  # Play the laser pulse after the Echo sequence
            # Measure and detect the dark counts on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times_dark, meas_len_1, counts_dark))
            save(counts_dark, counts_dark_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        counts_dark_st.buffer(len(t_vec)).average().save("counts_dark")
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
    job = qmm.simulate(config, hahn_echo, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    # execute QUA program
    job = qm.execute(hahn_echo)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts1", "counts2", "counts_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, counts_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(8 * t_vec, counts1 / 1000 / (meas_len_1 / u.s), label="x90_idle_x180_idle_x90")
        plt.plot(8 * t_vec, counts2 / 1000 / (meas_len_1 / u.s), label="x90_idle_x180_idle_-x90")
        plt.plot(8 * t_vec, counts_dark / 1000 / (meas_len_1 / u.s), label="dark counts")
        plt.xlabel("2 * idle time [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Hahn Echo")
        plt.legend()
        plt.pause(0.1)
