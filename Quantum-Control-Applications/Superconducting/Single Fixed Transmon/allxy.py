from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

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
                save(I, I_st[1])
                save(Q, Q_st[1])
            with case_(2):
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
                save(I, I_st[2])
                save(Q, Q_st[2])
            with case_(3):
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
                save(I, I_st[3])
                save(Q, Q_st[3])
            with case_(4):
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
                save(I, I_st[4])
                save(Q, Q_st[4])
            with case_(5):
                play("X/2", "qubit")
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
                play("Y/2", "qubit")
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
                save(I, I_st[7])
                save(Q, Q_st[7])
            with case_(8):
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
                save(I, I_st[8])
                save(Q, Q_st[8])
            with case_(9):
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
                save(I, I_st[9])
                save(Q, Q_st[9])
            with case_(10):
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
                save(I, I_st[10])
                save(Q, Q_st[10])
            with case_(11):
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
                save(I, I_st[11])
                save(Q, Q_st[11])
            with case_(12):
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
                save(I, I_st[12])
                save(Q, Q_st[12])
            with case_(13):
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
                save(I, I_st[13])
                save(Q, Q_st[13])
            with case_(14):
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
                save(I, I_st[14])
                save(Q, Q_st[14])
            with case_(15):
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
                save(I, I_st[15])
                save(Q, Q_st[15])
            with case_(16):
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
                save(I, I_st[16])
                save(Q, Q_st[16])
            with case_(17):
                play("X", "qubit")
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
                play("Y", "qubit")
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
                save(I, I_st[19])
                save(Q, Q_st[19])
            with case_(20):
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
                save(I, I_st[20])
                save(Q, Q_st[20])

    with stream_processing():
        for i in range(21):
            I_st[i].boolean_to_int().average().save(f"I{i}")
            Q_st[i].boolean_to_int().average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

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

sequence = [  # based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
    ("I", "I"),
    ("X", "X"),
    ("Y", "Y"),
    ("X", "Y"),
    ("Y", "X"),
    ("X/2", "I"),
    ("Y/2", "I"),
    ("X/2", "Y/2"),
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

plt.xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=90)
