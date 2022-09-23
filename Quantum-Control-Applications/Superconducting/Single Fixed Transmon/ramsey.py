"""
ramsey.py: Measures T2*
"""
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

tau_min = 4  # in clock cycles
tau_max = 100  # in clock cycles
d_tau = 2  # in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus

n_avg = 1e4
cooldown_time = 5 * qubit_T1 // 4

detuning = 1 * u.MHz  # in Hz

with program() as ramsey:
    n = declare(int)
    n_st = declare_stream()
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)

    update_frequency("qubit", qubit_IF + detuning)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus)):
            play("pi_half", "qubit")
            wait(tau, "qubit")
            play("pi_half", "qubit")
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
            wait(cooldown_time, "resonator")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)

    job = qm.execute(ramsey)
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
        plt.plot(4 * taus, I, ".", label="I")
        plt.plot(4 * taus, Q, ".", label="Q")
        plt.xlabel("Idle time [ns]")
        plt.ylabel("I & Q amplitude [a.u.]")
        plt.title("Ramsey")
        plt.legend()
        plt.pause(0.1)
