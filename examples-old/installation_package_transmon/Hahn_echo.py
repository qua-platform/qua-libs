from scipy import signal

from configuration import *
from qm.qua import *
from qm import SimulationConfig
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###############
# QUA program #
###############

tau_min = 4  # in units of clock cycles
tau_max = 200  # in units of clock cycles
dt = 10

cooldown_time = 50000 // 4

times = np.arange(tau_min, tau_max + dt / 2, dt)  # + dt/2 to add tau_max to times

N_max = 1000

detuning = 1e6  # detuning to qubit

with program() as ramsey:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    tau = declare(int)  # Variable delay between pi_half pulses
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    update_frequency("qubit", qubit_if - detuning)  # update qubit frequency with detuning

    # Pulse sequence
    ################
    with for_(n, 0, n < N_max, n + 1):

        with for_(tau, tau_min, tau < tau_max + dt / 2, tau + dt):
            wait(cooldown_time, "qubit")  # for qubit to decay
            play("pi_half", "qubit")  # pi_half
            wait(tau, "qubit")  # for evolution in the equator
            play("pi", "qubit")  # pi
            wait(tau, "qubit")  # for evolution in the equator
            play("pi_half", "qubit")  # pi_half
            align("resonator", "qubit")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(len(times)).average().save("I")
        Q_st.buffer(len(times)).average().save("Q")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=100000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, ramsey, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(ramsey)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    I_handle = res_handles.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handles.get("Q")
    Q_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            plt.title("Hahn echo")
            plt.plot(times * 4, I)
            plt.plot(times * 4, Q)
            plt.xlabel("Variable delay [ns]")
            plt.ylabel("demod signal [a.u.]")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    plt.title("Hahn echo")
    plt.plot(times * 4, I)
    plt.plot(times * 4, Q)
    plt.xlabel("Variable delay [ns]")
    plt.ylabel("demod signal [a.u.]")
