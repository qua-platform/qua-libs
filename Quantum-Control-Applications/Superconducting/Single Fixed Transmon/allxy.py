"""
allxy.py: Performs an ALLXY experiment to correct for gates imperfections
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details)
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
sequence = [  # based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]

n_points = 1e6

cooldown_time = 5 * qubit_T1 // 4

with program() as ALLXY:
    n = declare(int)
    r = Random()
    r_ = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = [declare_stream() for _ in range(21)]
    Q_st = [declare_stream() for _ in range(21)]

    with for_(n, 0, n < n_points, n + 1):
        assign(r_, r.rand_int(21))
        # Can replace by active reset
        wait(cooldown_time, "qubit")

        with switch_(r_):
            with case_(0):
                wait(x180_len // 4, "qubit")
                wait(x180_len // 4, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[0])
                save(Q, Q_st[0])
            with case_(1):
                play("x180", "qubit")
                play("x180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[1])
                save(Q, Q_st[1])
            with case_(2):
                play("y180", "qubit")
                play("y180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[2])
                save(Q, Q_st[2])
            with case_(3):
                play("x180", "qubit")
                play("y180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[3])
                save(Q, Q_st[3])
            with case_(4):
                play("y180", "qubit")
                play("x180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[4])
                save(Q, Q_st[4])
            with case_(5):
                play("x90", "qubit")
                wait(x180_len // 4, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[5])
                save(Q, Q_st[5])
            with case_(6):
                play("y90", "qubit")
                wait(x180_len // 4, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[6])
                save(Q, Q_st[6])
            with case_(7):
                play("x90", "qubit")
                play("y90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[7])
                save(Q, Q_st[7])
            with case_(8):
                play("y90", "qubit")
                play("x90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[8])
                save(Q, Q_st[8])
            with case_(9):
                play("x90", "qubit")
                play("y180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[9])
                save(Q, Q_st[9])
            with case_(10):
                play("y90", "qubit")
                play("x180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[10])
                save(Q, Q_st[10])
            with case_(11):
                play("x180", "qubit")
                play("y90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[11])
                save(Q, Q_st[11])
            with case_(12):
                play("y180", "qubit")
                play("x90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[12])
                save(Q, Q_st[12])
            with case_(13):
                play("x90", "qubit")
                play("x180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[13])
                save(Q, Q_st[13])
            with case_(14):
                play("x180", "qubit")
                play("x90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[14])
                save(Q, Q_st[14])
            with case_(15):
                play("y90", "qubit")
                play("y180", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[15])
                save(Q, Q_st[15])
            with case_(16):
                play("y180", "qubit")
                play("y90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[16])
                save(Q, Q_st[16])
            with case_(17):
                play("x180", "qubit")
                wait(x180_len // 4, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[17])
                save(Q, Q_st[17])
            with case_(18):
                play("y180", "qubit")
                wait(x180_len // 4, "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[18])
                save(Q, Q_st[18])
            with case_(19):
                play("x90", "qubit")
                play("x90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[19])
                save(Q, Q_st[19])
            with case_(20):
                play("y90", "qubit")
                play("y90", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st[20])
                save(Q, Q_st[20])

    with stream_processing():
        for i in range(21):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")


def allXY(pulses, index):
    if pulses[0] is not "I":
        play(pulses[0], "qubit")
    else:
        wait(x180_len // 4, "qubit")
    if pulses[1] is not "I":
        play(pulses[1], "qubit")
    else:
        wait(x180_len // 4, "qubit")
    align("qubit", "resonator")
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("cos", "out1", "sin", "out2", I),
        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
    )
    save(I, I_st[index])
    save(Q, Q_st[index])

with program() as ALLXY:
    n = declare(int)
    r = Random()
    r_ = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = [declare_stream() for _ in range(21)]
    Q_st = [declare_stream() for _ in range(21)]

    with for_(n, 0, n < n_points, n + 1):
        assign(r_, r.rand_int(21))
        # Can replace by active reset
        wait(cooldown_time, "qubit")

        # with switch_(r_):
        #     for i in range(len(sequence)):
        #         with case_(i):
        #             allXY(sequence[i], i)

        with switch_(r_):
            with case_(0):
                allXY(sequence[0], 0)
            with case_(1):
                allXY(sequence[1], 1)
            with case_(2):
                allXY(sequence[2], 2)
            with case_(3):
                allXY(sequence[3], 3)
            with case_(4):
                allXY(sequence[4], 4)
            with case_(5):
                allXY(sequence[5], 5)
            with case_(6):
                allXY(sequence[6], 6)
            with case_(7):
                allXY(sequence[7], 7)
            with case_(8):
                allXY(sequence[8], 8)
            with case_(9):
                allXY(sequence[9], 9)
            with case_(10):
                allXY(sequence[10], 10)
            with case_(11):
                allXY(sequence[11], 11)
            with case_(12):
                allXY(sequence[12], 12)
            with case_(13):
                allXY(sequence[13], 13)
            with case_(14):
                allXY(sequence[14], 14)
            with case_(15):
                allXY(sequence[15], 15)
            with case_(16):
                allXY(sequence[16], 16)
            with case_(17):
                allXY(sequence[17], 17)
            with case_(18):
                allXY(sequence[18], 18)
            with case_(19):
                allXY(sequence[19], 19)
            with case_(20):
                allXY(sequence[20], 20)

    with stream_processing():
        for i in range(21):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, ALLXY, simulation_config)
    job.get_simulated_samples().con1.plot()

else:

    qm = qmm.open_qm(config)

    job = qm.execute(ALLXY)
    job.result_handles.wait_for_all_values()

    I = []
    Q = []
    for x in range(21):
        I.append(job.result_handles.get(f"I{x}").fetch_all())
        Q.append(job.result_handles.get(f"Q{x}").fetch_all())

    I = np.array(I)
    Q = np.array(Q)

    plt.figure()
    plt.plot(I)

    plt.figure()
    plt.plot(Q)

    plt.xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=90)
