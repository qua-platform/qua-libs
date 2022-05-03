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

# macros
#########


def discriminator():

    Idis = declare(fixed)
    Qdis = declare(fixed)
    I_th = declare(fixed, value=0.0)  # threshold value for discrimination
    Q_th = declare(fixed, value=0.0)  # threshold value for discrimination
    st = declare(int)

    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Idis),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Qdis),
    )

    with if_(Idis > I_th):
        assign(st, 1)
    with else_():
        assign(st, -1)

    return st


def active_reset():

    Idis = declare(fixed)
    Qdis = declare(fixed)
    I_th = declare(fixed, value=0.0)  # threshold value for discrimination

    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Idis),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Qdis),
    )

    play("pi", "qubit", condition=Idis > I_th)


n_avg = 1000  # number of averages

cooldown_time = 50000 // 4  # qubit decay time

with program() as iq_blobs:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    Ig_st = declare_stream()  # stream for Ig
    Qg_st = declare_stream()  # stream for Qg
    Ie_st = declare_stream()  # stream for Ie
    Qe_st = declare_stream()  # stream for Qe
    state = declare(int)

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        # ground IQ blob
        wait(cooldown_time, "qubit")  # wait for qubit to decay
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("cos", "out1", "sin", "out2", I),
            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
        )
        save(I, Ig_st)
        save(Q, Qg_st)

        align()  # global align

        # excited IQ blob
        wait(cooldown_time, "qubit")  # wait for qubit to decay
        play("pi", "qubit")  # to populate |e> state
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("cos", "out1", "sin", "out2", I),
            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
        )
        save(I, Ie_st)
        save(Q, Qe_st)

        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        # save all to generate blobs
        Ig_st.save_all("Ig")
        Qg_st.save_all("Qg")
        Ie_st.save_all("Ie")
        Qe_st.save_all("Qe")
        # mean values
        Ig_st.average().save("Ig.avg")
        Qg_st.average().save("Qg.avg")
        Ie_st.average().save("Ie.avg")
        Qe_st.average().save("Qe.avg")
        # variances
        (((Ig_st * Ig_st).average()) - (Ig_st.average() * Ig_st.average())).save("Igvar")
        (((Qg_st * Qg_st).average()) - (Qg_st.average() * Qg_st.average())).save("Qgvar")
        (((Ie_st * Ie_st).average()) - (Ie_st.average() * Ie_st.average())).save("Ievar")
        (((Qe_st * Qe_st).average()) - (Qe_st.average() * Qe_st.average())).save("Qevar")


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
    job = qmm.simulate(config, iq_blobs, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(iq_blobs)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    Ig_handle = res_handles.get("Ig")
    Ig_handle.wait_for_values(1)
    Qg_handle = res_handles.get("Qg")
    Qg_handle.wait_for_values(1)
    Ie_handle = res_handles.get("Ie")
    Ie_handle.wait_for_values(1)
    Qe_handle = res_handles.get("Qe")
    Qe_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            Ig = Ig_handle.fetch_all()
            Qg = Qg_handle.fetch_all()
            Ie = Ie_handle.fetch_all()
            Qe = Qe_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            plt.title("IQ blobs")
            plt.plot(Ig, Qg, ".")
            plt.plot(Ie, Qe, ".")
            plt.axis("equal")
            plt.axhline(y=0)
            plt.axvline(x=0)
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    Ig = Ig_handle.fetch_all()
    Qg = Qg_handle.fetch_all()
    Ie = Ie_handle.fetch_all()
    Qe = Qe_handle.fetch_all()
    plt.title("IQ blobs")
    plt.plot(Ig, Qg, ".")
    plt.plot(Ie, Qe, ".")
    plt.axis("equal")
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel("I")
    plt.ylabel("Q")

    Ig_avg = res_handles.get("Ig.avg").fetch_all()
    Qg_avg = res_handles.get("Qg.avg").fetch_all()
    Ie_avg = res_handles.get("Ie.avg").fetch_all()
    Qe_avg = res_handles.get("Qe.avg").fetch_all()

    Ig_var = res_handles.get("Igvar").fetch_all()
    Qg_var = res_handles.get("Qgvar").fetch_all()
    Ie_var = res_handles.get("Ievar").fetch_all()
    Qe_var = res_handles.get("Qevar").fetch_all()

    Ig_std = np.sqrt(Ig_var)
    Qg_std = np.sqrt(Qg_var)
    Ie_std = np.sqrt(Ie_var)
    Qe_std = np.sqrt(Qg_var)

    print("|g> I-Q values are:", Ig_avg, Qg_avg, "and stds are:", Ig_std, Qg_std)
    print("|e> I-Q values are:", Ie_avg, Qe_avg, "and stds are:", Ig_std, Qg_std)
