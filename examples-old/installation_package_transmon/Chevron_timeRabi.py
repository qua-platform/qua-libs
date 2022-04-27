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

t_min = 4
t_max = 200
dt = 100

times = np.arange(t_min, t_max + dt / 2, dt)

f_min = 30e6
f_max = 60e6
df = 10e6

freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

N_max = 1000

cooldown_time = 50000 // 4

with program() as chevron_rabi:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    tau = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # Pulse sequence
    ################
    with for_(n, 0, n < N_max, n + 1):

        with for_(tau, t_min, tau < t_max + dt / 2, tau + dt):
            with for_(f, f_min, f <= f_max, f + df):
                update_frequency("qubit", f)  # update frequency of the qubit
                wait(cooldown_time, "qubit")  # wait for qubit to decay
                play("pi", "qubit", duration=tau)
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
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(len(times), len(freqs)).average().save("I")
        Q_st.buffer(len(times), len(freqs)).average().save("Q")

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
    job = qmm.simulate(config, chevron_rabi, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(chevron_rabi)  # execute QUA program

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
            plt.title("Chevron Time Rabi")
            plt.pcolor(freqs, times, I)
            plt.xlabel("IF [Hz]")
            plt.ylabel("Variable pulse length")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    plt.title("Chevron Time Rabi")
    plt.pcolor(freqs, times, I)
    plt.xlabel("IF [Hz]")
    plt.ylabel("Variable pulse length")
