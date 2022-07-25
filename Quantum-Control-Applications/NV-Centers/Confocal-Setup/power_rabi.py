"""
power_rabi.py: A Rabi experiment sweeping the amplitude of the MW pulse.
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

a_min = 0.1  # proportional factor to the pulse amplitude
a_max = 1  # proportional factor to the pulse amplitude
da = 0.02
a_vec = np.arange(a_min, a_max + da / 2, da)  # +da/2 to include a_max
n_avg = 1e6  # number of iterations

with program() as power_rabi:
    counts = declare(int)  # variable for number of counts
    times = declare(int, size=100)
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    counts_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    play("laser_ON", "AOM")
    wait(100, "AOM")
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, a_vec)):
            play("pi" * amp(a), "NV")  # pulse of varied amplitude
            align()
            play("laser_ON", "AOM")
            measure("readout", "SPCM", None, time_tagging.analog(times, meas_len, counts))
            save(counts, counts_st)  # save counts
            wait(100)

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(a_vec)).average().save("counts")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True
if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    # execute QUA program
    job = qm.execute(power_rabi)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(a_vec * pi_amp_NV, counts / 1000 / (meas_len * 1e-9))
        plt.xlabel("Amplitude [volts]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Power Rabi")
        plt.pause(0.1)
