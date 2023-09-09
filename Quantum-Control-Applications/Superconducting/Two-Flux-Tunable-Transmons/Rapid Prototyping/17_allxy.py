"""
allxy.py: Performs an ALLXY experiment to correct for gates imperfections
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details)
"""
#%%
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].qubit_name + "_z"
q2_z = machine.qubits[active_qubits[1]].qubit_name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2
#%%
##############################
# Program-specific variables #
##############################

n_points = 100000
cooldown_time = 5 * max(qb1.T1, qb2.T1)

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
        play(pulses[0], qubit.qubit_name + "_xy")  # Either play the sequence
    else:
        wait(qubit.xy.pi_length // 4, qubit.qubit_name + "_xy")  # or wait if sequence is identity
    if pulses[1] != "I":
        play(pulses[1], qubit.qubit_name + "_xy")  # Either play the sequence
    else:
        wait(qubit.xy.pi_length // 4, qubit.qubit_name + "_xy")  # or wait if sequence is identity

    align()
    # Play the readout on the other resonator to measure in the same condition as when optimizing readout
    if resonator == rr1:
        measure("readout", rr2.resonator_name, None)
    else:
        measure("readout", rr1.resonator_name, None)
    measure(
        "readout",
        resonator.resonator_name,
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_xy),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_xy),
    )
    return I_xy, Q_xy


###################
# The QUA program #
###################
def get_prog(qubit, resonator):
    with program() as ALLXY:
        n = declare(int)
        n_st = declare_stream()
        r = Random()
        r_ = declare(int)
        I_st = [declare_stream() for _ in range(21)]
        Q_st = [declare_stream() for _ in range(21)]

        # Bring the active qubits to the maximum frequency point
        set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
        set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

        with for_(n, 0, n < n_points, n + 1):
            save(n, n_st)
            assign(r_, r.rand_int(21))
            # Can replace by active reset
            wait(cooldown_time * u.ns)
            # Plays a random XY sequence
            with switch_(r_):
                for i in range(21):
                    with case_(i):
                        I, Q = allXY(sequence[i], qubit, resonator)
                        save(I, I_st[i])
                        save(Q, Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(21):
                I_st[i].average().save(f"I{i}")
                Q_st[i].average().save(f"Q{i}")
    return ALLXY

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=50000)  # in clock cycles
    job = qmm.simulate(config, ALLXY, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(config)

    for qb, rr in [[qb1, rr1],[qb2, rr2]]:
        job = qm.execute(get_prog(qb, rr))

        fig, ax = plt.subplots(2, 1)
        interrupt_on_close(fig, job)
        data_list = ["n"] + np.concatenate([[f"I{i}", f"Q{i}"] for i in range(21)]).tolist()
        results = fetching_tool(job, data_list, mode="wait_for_all")
        # while results.is_processing():
        res = results.fetch_all()
        I = np.array(res[1::2])
        Q = np.array(res[2::2])

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
        plt.suptitle(f"All XY {qb.qubit_name}")
        plt.tight_layout()
        plt.pause(1.0)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# %%
