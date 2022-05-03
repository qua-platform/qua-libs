from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

n_avg = 1000  # number of averages

cooldown_time = 50000 // 4  # qubit decay time

p_min = 0
p_max = 8
dp = 2

ps = np.arange(p_min, p_max + dp / 2, dp)

a_min = 0.8
a_max = 1.2
da = 0.1

amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as power_rabi:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    p = declare(int)  # variable to loop over number of pulses
    a = declare(fixed)  # variable for amps sweep
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):  # averaging QUA for_ loop

        with for_(a, a_min, a < a_max + da / 2, a + da):  # to tweak around the amplitude

            with for_(p, p_min, p < p_max + dp / 2, p + dp):  # QUA for_, iterates over number (odd) of pulses
                with switch_(p):  # QUA switch_
                    for i in range(p_min, p_max + dp, dp):  # Python for
                        with case_(i):  # QUA case_
                            wait(cooldown_time, "qubit")  # wait for qubit to decay
                            for j in range(i):  # Python for
                                play("pi" * amp(a), "qubit")
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
        I_st.buffer(len(amps), len(ps)).average().save("I")
        Q_st.buffer(len(amps), len(ps)).average().save("Q")

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
    job = qmm.simulate(config, power_rabi, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(power_rabi)  # execute QUA program

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
            plt.title("Pi calibration")
            plt.pcolor(ps, amps, I)
            plt.xlabel("Iteration of pulses")
            plt.ylabel("Proportional amplitude")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    plt.title("Pi calibration")
    plt.pcolor(ps, amps, I)
    plt.xlabel("Iteration of pulses")
    plt.ylabel("Proportional amplitude")
