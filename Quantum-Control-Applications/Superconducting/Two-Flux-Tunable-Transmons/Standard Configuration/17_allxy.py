from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter

"""
allxy.py: Performs an ALLXY experiment to correct for gates imperfections
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details)
"""

##############################
# Program-specific variables #
##############################
qb = "q1_xy"
res = "rr1"
n_points = 1000
cooldown_time = 10_000

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
def allXY(pulses, qubit, resonator):
    """
    Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
    **Example:** I, Q = allXY(['I', 'y90'])
    :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
        operations in the config except for the identity operation that must be called 'I'.
    :param qubit: The qubit element as defined in the config.
    :param resonator: The resonator element as defined in the config.
    :return: two QUA variables for the 'I' and 'Q' quadratures measured after the sequence.
    """
    I_xy = declare(fixed)
    Q_xy = declare(fixed)
    if pulses[0] != "I":
        play(pulses[0], qubit)  # Either play the sequence
    else:
        wait(pi_len // 4, qubit)  # or wait if sequence is identity
    if pulses[1] != "I":
        play(pulses[1], qubit)  # Either play the sequence
    else:
        wait(pi_len // 4, qubit)  # or wait if sequence is identity

    align(qubit, resonator)
    # Play through the 2nd resonator to be in the same condition as when the readout was optimized
    if resonator == "rr1":
        align(qubit, "rr2")
        measure("readout", "rr2", None)
    elif resonator == "rr2":
        align(qubit, "rr1")
        measure("readout", "rr1", None)
    measure(
        "readout",
        resonator,
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_xy),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_xy),
    )
    return I_xy, Q_xy


###################
# The QUA program #
###################
with program() as ALLXY:
    n = declare(int)
    n_st = declare_stream()
    r = Random()
    r_ = declare(int)
    I_st = [declare_stream() for _ in range(21)]
    Q_st = [declare_stream() for _ in range(21)]

    with for_(n, 0, n < n_points, n + 1):
        save(n, n_st)
        assign(r_, r.rand_int(21))
        # Can replace by active reset
        wait(cooldown_time * u.ns, qb)
        # Plays a random XY sequence
        with switch_(r_):
            for i in range(21):
                with case_(i):
                    I, Q = allXY(sequence[i], qb, res)
                    save(I, I_st[i])
                    save(Q, Q_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(21):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=50000)  # in clock cycles
    job = qmm.simulate(config, ALLXY, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:

    qm = qmm.open_qm(config)

    job = qm.execute(ALLXY)

    fig, ax = plt.subplots(2, 1)
    interrupt_on_close(fig, job)
    data_list = ["n"] + np.concatenate([[f"I{i}", f"Q{i}"] for i in range(21)]).tolist()
    results = fetching_tool(job, data_list, mode="live")
    while results.is_processing():
        res = results.fetch_all()
        I = np.array(res[1::2])
        Q = np.array(res[2::2])
        n = res[0]
        progress_counter(n, n_points, start_time=results.start_time)

        ax[0].cla()
        ax[0].plot(-I, "-*")
        ax[0].plot([np.max(-I)] * 5 + [(np.mean(-I))] * 12 + [np.min(-I)] * 4, "-")
        ax[0].set_ylabel("I quadrature [a.u.]")
        ax[0].set_xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=45)
        ax[1].cla()
        ax[1].plot(-I, "-*")
        ax[1].plot([np.max(-I)] * 5 + [(np.mean(-I))] * 12 + [np.min(-I)] * 4, "-")
        ax[1].set_ylabel("Q quadrature [a.u.]")
        ax[1].set_xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=45)
        plt.suptitle("All XY (n: %s)" % (n))
        plt.tight_layout()
        plt.pause(1.0)

    plt.show()
