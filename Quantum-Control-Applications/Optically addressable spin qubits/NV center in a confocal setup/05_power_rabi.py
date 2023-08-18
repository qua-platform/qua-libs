"""
power_rabi.py: A Rabi experiment sweeping the amplitude of the MW pulse.
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

a_vec = np.arange(0.1, 1, 0.02)  # +da/2 to include a_max
n_avg = 1_000_000  # number of iterations

with program() as power_rabi:
    counts = declare(int)  # variable for number of counts
    times = declare(int, size=100)
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM1")
    wait(100 * u.ns, "AOM1")

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, a_vec)):
            play("x180" * amp(a), "NV")  # pulse of varied amplitude
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts
            wait(100 * u.ns)  # wait in between iterations

            align()

            play("x180" * amp(0), "NV")
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_dark_st)  # save counts
            wait(100 * u.ns)  # wait in between iterations

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(a_vec)).average().save("counts")
        counts_dark_st.buffer(len(a_vec)).average().save("counts_dark")
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
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(power_rabi)
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
        plt.plot(a_vec * pi_amp_NV, counts / 1000 / (meas_len_1 * 1e-9))
        plt.plot(a_vec * pi_amp_NV, counts_dark / 1000 / (meas_len_1 * 1e-9))
        plt.xlabel("Amplitude [volts]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Power Rabi")
        plt.pause(0.1)
