"""
ramsey.py: Measures T2*.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

t_min = 16 // 4  # in clock cycles units (must be >= 4)
t_max = 1000 // 4  # in clock cycles units
dt = 40 // 4  # in clock cycles units
t_vec = np.arange(t_min, t_max + 0.1, dt)  # +0.1 to include t_max in array
n_avg = 1e6

with program() as ramsey:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)
    times2 = declare(int, size=100)
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM")
    wait(100)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            play("pi_half", "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            play("pi_half", "NV")  # Pi/2 pulse to qubit

            align()

            play("laser_ON", "AOM")
            measure("readout", "SPCM", None, time_tagging.analog(times1, meas_len, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(100, "AOM")

            align()

            play("pi_half", "NV")  # Pi/2 pulse to qubit
            wait(t, "NV")  # variable delay in spin Echo
            frame_rotation_2pi(0.5, "NV")  # Turns next pulse to -x
            play("pi_half", "NV")  # Pi/2 pulse to qubit
            reset_frame("NV")

            align()

            play("laser_ON", "AOM")
            measure("readout", "SPCM", None, time_tagging.analog(times2, meas_len, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(100, "AOM")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True
if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    # execute QUA program
    job = qm.execute(ramsey)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts1", "counts2", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(4 * t_vec, counts1 / 1000 / (meas_len / u.s), counts2 / 1000 / (meas_len / u.s))
        plt.xlabel("Dephasing time [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.legend(("counts 1", "counts 2"))
        plt.title("Ramsey")
        plt.pause(0.1)
