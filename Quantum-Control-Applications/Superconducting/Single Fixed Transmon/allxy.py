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


##############################
# Program-specific variables #
##############################
n_points = 1e2
cooldown_time = 5 * qubit_T1 // 4

# All XY sequences. The sequence names must match corresponding operation in the config
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
# All XY macro generating the pulse sequences from a python list.
def allXY(pulses):
    """
    Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
    **Example:** I, Q = allXY(['I', 'y90'])

    :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
        operations in the config except for the identity operation that must be called 'I'.
    :return: two QUA variables for the 'I' and 'Q' quadratures measured after the sequence.
    """
    I_xy = declare(fixed)
    Q_xy = declare(fixed)
    if pulses[0] != "I":
        play(pulses[0], "qubit")  # Either play the sequence
    else:
        wait(x180_len // 4, "qubit")  # or wait if sequence is identity
    if pulses[1] != "I":
        play(pulses[1], "qubit")  # Either play the sequence
    else:
        wait(x180_len // 4, "qubit")  # or wait if sequence is identity
    align("qubit", "resonator")
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("cos", "out1", "sin", "out2", I_xy),
        dual_demod.full("minus_sin", "out1", "cos", "out2", Q_xy),
    )
    return I_xy, Q_xy


###################
# The QUA program #
###################
with program() as ALLXY:
    n = declare(int)
    r = Random()
    r_ = declare(int)
    I_st = [declare_stream() for _ in range(21)]
    Q_st = [declare_stream() for _ in range(21)]

    with for_(n, 0, n < n_points, n + 1):
        assign(r_, r.rand_int(21))
        # Can replace by active reset
        wait(cooldown_time, "qubit")
        # Plays a random XY sequence
        with switch_(r_):
            for i in range(len(sequence)):
                with case_(i):
                    I, Q = allXY(sequence[i])
                    save(I, I_st[i])
                    save(Q, Q_st[i])

    with stream_processing():
        for i in range(21):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=100000)  # in clock cycles
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
    ax1 = plt.subplot(211)
    ax1.plot(I)
    ax1.set_ylabel("I quadrature [a.u.]")
    ax1.set_xticklabels("")
    ax2 = plt.subplot(212)
    ax2.plot(Q)
    ax2.set_ylabel("Q quadrature [a.u.]")
    ax2.set_xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=45)
    plt.suptitle("All XY")
    plt.tight_layout()
