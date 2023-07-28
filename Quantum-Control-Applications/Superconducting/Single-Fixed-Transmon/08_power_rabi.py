from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_avg = 10000

cooldown_time = 5 * qubit_T1

a_min = 0.0
a_max = 1.0
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes
err_amp = 1  # Number of played qubit pulses for getting a better estimate of the pi amplitude

with program() as power_rabi:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            # Loop for error amplification (perform many qubit pulses)
            for i in range(err_amp):
                play("x180" * amp(a), "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(cooldown_time * u.ns, "resonator")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)

    job = qm.execute(power_rabi)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.cla()
        plt.plot(amps * x180_amp, I, ".", label="I")
        plt.plot(amps * x180_amp, Q, ".", label="Q")
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("I & Q amplitude [a.u.]")
        plt.legend()
        plt.pause(0.1)
