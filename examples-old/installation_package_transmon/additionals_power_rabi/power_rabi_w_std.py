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

n_avg = 10000  # number of averages

cooldown_time = 50000 // 4  # qubit decay time

a_min = 0.0
a_max = 1.0
da = 0.1

amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as power_rabi:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    a = declare(fixed)  # variable for amps sweep
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        with for_(
            a, a_min, a < a_max + da / 2, a + da
        ):  # Notice it's + da/2 to include a_max (This is only for fixed!)
            wait(cooldown_time, "qubit")  # wait for qubit to decay
            play("gaussian" * amp(a), "qubit")  # play gaussian pulse with variable amplitude
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            # assign(I, I << 1)  # if I measured is too small bitshift is needed
            # assign(Q, Q << 1)  # if Q measured is too small bitshift is needed
            save(I, I_st)
            save(Q, Q_st)
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        # mean values
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")
        # variances
        (
            ((I_st.buffer(len(amps)) * I_st.buffer(len(amps))).average())
            - (I_st.buffer(len(amps)).average() * I_st.buffer(len(amps)).average())
        ).save("Ivar")
        (
            ((Q_st.buffer(len(amps)) * Q_st.buffer(len(amps))).average())
            - (Q_st.buffer(len(amps)).average() * Q_st.buffer(len(amps)).average())
        ).save("Qvar")

#######################
# Simulate or execute #
#######################

simulate = False

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
    Ivar_handle = res_handles.get("Ivar")
    Ivar_handle.wait_for_values(1)
    Qvar_handle = res_handles.get("Qvar")
    Qvar_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            Ivar = Ivar_handle.fetch_all()
            Qvar = Qvar_handle.fetch_all()
            Istd = np.sqrt(Ivar)
            Qstd = np.sqrt(Qvar)
            iteration = iteration_handle.fetch_all() + 1
            plt.title("Power Rabi")
            plt.errorbar(
                amps,
                I,
                yerr=Istd,
                uplims=True,
                lolims=True,
                label="uplims=True, lolims=True",
            )
            plt.errorbar(
                amps,
                Q,
                yerr=Qstd,
                uplims=True,
                lolims=True,
                label="uplims=True, lolims=True",
            )
            plt.xlabel("Amps")
            plt.ylabel("demod signal [a.u.]")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    Ivar = Ivar_handle.fetch_all()
    Qvar = Qvar_handle.fetch_all()
    Istd = np.sqrt(Ivar)
    Qstd = np.sqrt(Qvar)
    plt.title("Power Rabi")
    plt.errorbar(amps, I, yerr=Istd, uplims=True, lolims=True, label="uplims=True, lolims=True")
    plt.errorbar(amps, Q, yerr=Qstd, uplims=True, lolims=True, label="uplims=True, lolims=True")
    plt.xlabel("Amps")
    plt.ylabel("demod signal [a.u.]")
