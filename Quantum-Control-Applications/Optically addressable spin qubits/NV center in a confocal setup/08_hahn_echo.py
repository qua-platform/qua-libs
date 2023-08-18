"""
hahn_echo.py: Measures T2.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

t_vec = np.arange(4, 500, 20)  # +0.1 to include t_max in array
n_avg = 1_000_000

with program() as hahn_echo:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    counts2_dark = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)
    times2 = declare(int, size=100)
    times2_dark = declare(int, size=100)
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    counts_2_st_dark = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM1")
    wait(100 * u.ns)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("x180" * amp(1), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit

            align()

            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times1, meas_len_1, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(100 * u.ns, "AOM1")

            align()

            play("x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("x180" * amp(1), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("-x90" * amp(1), "NV")  # Pi/2 pulse to qubit
            reset_frame("NV")

            align()

            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2_dark, meas_len_1, counts2_dark))
            save(counts2, counts_2_st)  # save counts
            wait(100 * u.ns, "AOM1")

            align()

            play("x90" * amp(0), "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("x180" * amp(0), "NV")  # Pi pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("-x90" * amp(0), "NV")  # Pi/2 pulse to qubit
            reset_frame("NV")

            align()

            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2, meas_len_1, counts2))
            save(counts2_dark, counts_2_st_dark)  # save counts
            wait(100 * u.ns, "AOM1")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        counts_2_st_dark.buffer(len(t_vec)).average().save("counts2_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

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
    results = fetching_tool(job, data_list=["counts1", "counts2", "counts2_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, counts2_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(8 * t_vec, counts1 / 1000 / (meas_len_1 / u.s))
        plt.plot(8 * t_vec, counts2 / 1000 / (meas_len_1 / u.s))
        plt.plot(8 * t_vec, counts2_dark / 1000 / (meas_len_1 / u.s))
        plt.xlabel("2*tau [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Hahn Echo")
        plt.pause(0.1)

