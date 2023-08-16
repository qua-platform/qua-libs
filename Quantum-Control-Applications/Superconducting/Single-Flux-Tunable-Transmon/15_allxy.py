from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig

"""
Performs an ALLXY experiment to estimate gates imperfection
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details)
"""

##############################
# Program-specific variables #
##############################
n_points = 1e6
cooldown_time = 5 * qubit_T1

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
    I_st = [declare_stream() for _ in range(len(sequence))]
    Q_st = [declare_stream() for _ in range(len(sequence))]
    n_st = declare_stream()

    with for_(n, 0, n < n_points, n + 1):
        assign(r_, r.rand_int(len(sequence)))
        # Can replace by active reset
        wait(cooldown_time * u.ns, "qubit")
        # Plays a random XY sequence
        with switch_(r_):
            for i in range(len(sequence)):
                with case_(i):
                    I, Q = allXY(sequence[i])
                    save(I, I_st[i])
                    save(Q, Q_st[i])
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        for i in range(len(sequence)):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=100000)  # in clock cycles
    job = qmm.simulate(config, ALLXY, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(ALLXY)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    data_list = ["iteration"] + list(np.concatenate([[f"I{i}", f"Q{i}"] for i in range(len(sequence))]))
    results = fetching_tool(job, data_list, mode="live")
    while results.is_processing():
        res = results.fetch_all()
        I = np.array(res[1::2])
        Q = np.array(res[2::2])
        n = res[0]
        progress_counter(n, n_points, start_time=results.start_time)
        plt.subplot(211)
        plt.cla()
        plt.plot(-I)
        plt.plot([np.max(-I)] * 5 + [(np.mean(-I))] * 12 + [np.min(-I)] * 4, "-")
        plt.ylabel("I quadrature [a.u.]")
        plt.xticks(ticks=range(len(sequence)), labels=["" for _ in sequence], rotation=45)
        plt.subplot(212)
        plt.cla()
        plt.plot(-Q)
        plt.plot([np.max(-Q)] * 5 + [(np.mean(-Q))] * 12 + [np.min(-Q)] * 4, "-")
        plt.ylabel("Q quadrature [a.u.]")
        plt.xticks(ticks=range(len(sequence)), labels=[str(el) for el in sequence], rotation=45)
        plt.suptitle("All XY")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()