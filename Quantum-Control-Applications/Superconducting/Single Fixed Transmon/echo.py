"""
Measures T2
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig

###################
# The QUA program #
###################

tau_min = 4  # in clock cycles
tau_max = 100  # in clock cycles
dtau = 2  # in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, dtau)  # + 0.1 to add tau_max to taus

n_avg = 1e4
cooldown_time = 5 * qubit_T1 // 4

with program() as echo:
    n = declare(int)
    n_st = declare_stream()
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's <= to include t_max (This is only for integers!)
        with for_(tau, tau_min, tau <= tau_max, tau + dtau):
            play("x90", "qubit")
            wait(tau, "qubit")
            play("x180", "qubit")
            wait(tau, "qubit")
            play("x90", "qubit")
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

######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, echo, simulation_config)
    job.get_simulated_samples().con1.plot()

else:

    qm = qmm.open_qm(config)
    job = qm.execute(echo)
    res_handles = job.result_handles
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    iteration_handle = res_handles.get("iteration")
    I_handle.wait_for_values(1)
    Q_handle.wait_for_values(1)
    iteration_handle.wait_for_values(1)
    next_percent = 0.1  # First time print 10%


    def on_close(event):
        event.canvas.stop_event_loop()
        job.halt()


    f = plt.figure()
    f.canvas.mpl_connect("close_event", on_close)
    print("Progress =", end=" ")

    while res_handles.is_processing():
        plt.cla()
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        iteration = iteration_handle.fetch_all()
        if iteration / n_avg > next_percent:
            percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
            print(f"{percent}%", end=" ")
            next_percent = percent / 100 + 0.1  # Print every 10%

        plt.plot(2 * taus, I, ".", label="I")
        plt.plot(2 * taus, Q, ".", label="Q")

        plt.legend()
        plt.pause(0.1)

    plt.cla()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    print(f"{round(iteration/n_avg * 100)}%")
    plt.plot(2 * taus, I, ".", label="I")
    plt.plot(2 * taus, Q, ".", label="Q")

    plt.legend()
