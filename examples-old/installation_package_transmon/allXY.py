from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import numpy as np
from matplotlib import pyplot as plt
from qm.qua import *
from qm import LoopbackInterface

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

cooldown_time = 50000 // 4  # decay time for qubit

with program() as allxy:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    r = Random()  # variable to generate QUA random
    n_st = declare_stream()  # to save iteration
    r_ = declare(int)  # variable to assign random
    st = [declare_stream() for i in range(21)]  # list of streams to save measure()
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    with for_(n, 0, n < n_avg, n + 1):

        assign(r_, r.rand_int(21))

        with switch_(r_):
            with case_(0):
                wait(cooldown_time, "qubit")
                wait(pi_len, "qubit")
                wait(pi_len, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[0])
            with case_(1):
                wait(cooldown_time, "qubit")
                play("X", "qubit")
                play("X", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[1])
            with case_(2):
                wait(cooldown_time, "qubit")
                play("Y", "qubit")
                play("Y", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[2])
            with case_(3):
                wait(cooldown_time, "qubit")
                play("X", "qubit")
                play("Y", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[3])
            with case_(4):
                wait(cooldown_time, "qubit")
                play("Y", "qubit")
                play("X", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[4])
            with case_(5):
                wait(cooldown_time, "qubit")
                play("X/2", "qubit")
                wait(pi_len, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[5])
            with case_(6):
                wait(cooldown_time, "qubit")
                play("Y/2", "qubit")
                wait(pi_len, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[6])
            with case_(7):
                wait(cooldown_time, "qubit")
                play("X/2", "qubit")
                play("Y/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[7])
            with case_(8):
                wait(cooldown_time, "qubit")
                play("Y/2", "qubit")
                play("X/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[8])
            with case_(9):
                wait(cooldown_time, "qubit")
                play("X/2", "qubit")
                play("Y", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[9])
            with case_(10):
                wait(cooldown_time, "qubit")
                play("Y/2", "qubit")
                play("X", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[10])
            with case_(11):
                wait(cooldown_time, "qubit")
                play("X", "qubit")
                play("Y/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[11])
            with case_(12):
                wait(cooldown_time, "qubit")
                play("Y", "qubit")
                play("X/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[12])
            with case_(13):
                wait(cooldown_time, "qubit")
                play("X/2", "qubit")
                play("X", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[13])
            with case_(14):
                wait(cooldown_time, "qubit")
                play("X", "qubit")
                play("X/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[14])
            with case_(15):
                wait(cooldown_time, "qubit")
                play("Y/2", "qubit")
                play("Y", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[15])
            with case_(16):
                wait(cooldown_time, "qubit")
                play("Y", "qubit")
                play("Y/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[16])
            with case_(17):
                wait(cooldown_time, "qubit")
                play("X", "qubit")
                wait(pi_len, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[17])
            with case_(18):
                wait(cooldown_time, "qubit")
                play("Y", "qubit")
                wait(pi_len, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[18])
            with case_(19):
                wait(cooldown_time, "qubit")
                play("X/2", "qubit")
                play("X/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[19])
            with case_(20):
                wait(cooldown_time, "qubit")
                play("Y/2", "qubit")
                play("Y/2", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(Q, st[20])

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        for i in range(21):
            st[i].average().save("res{}".format(i))

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
    job = qmm.simulate(config, allxy, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:

    job = qm.execute(allxy)  # execute QUA program

    res_handles = job.result_handles  # get access to handles

    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    sequence = [  # based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
        ("I", "I"),
        ("X", "X"),
        ("Y", "Y"),
        ("X", "Y"),
        ("Y", "X"),
        ("X/2", "I"),
        ("Y/2", "I"),
        ("X/2", "Y"),
        ("Y/2", "X/2"),
        ("X/2", "Y"),
        ("Y/2", "X"),
        ("X", "Y/2"),
        ("Y", "X/2"),
        ("X/2", "X"),
        ("X", "X/2"),
        ("Y/2", "Y"),
        ("Y", "Y/2"),
        ("X", "I"),
        ("Y", "I"),
        ("X/2", "X/2"),
        ("Y/2", "Y/2"),
    ]

    for x in range(21):
        res_handles.get("res{}".format(x)).wait_for_values(1)

    plt.figure()

    while res_handles.is_processing():
        res = []
        try:
            for x in range(21):
                res.append(job.result_handles.get("res{}".format(x)).fetch_all())
            iteration = iteration_handle.fetch_all()
            print(iteration)
            plt.xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=90)
            res = np.array(res)
            values = (((res - res[0]) / res[-1]) * (-2)) + 1  # minus the ground state value and
            # divided by the excited state value [0,1)
            # values = res * (-2) + 1
            plt.plot(values, ".")
            # plt.plot(values / values[0], '.')
            plt.pause(0.1)
            plt.clf()

        except Exception as e:
            pass

    res = []

    for x in range(21):
        res.append(job.result_handles.get("res{}".format(x)).fetch_all())

    plt.xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=90)
    res = np.array(res)
    values = (((res - res[0]) / res[-1]) * (-2)) + 1  # minus the ground state value and
    # divided by the excited state value [0,1)
    # values = res * (-2) + 1
    plt.plot(values, ".")
    # plt.plot(values / values[0], '.')
