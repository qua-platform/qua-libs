from qm.qua import *
from qualang_tools.bakery.bakery import Baking

from quam_libs.experiments.two_qubit_rb import TwoQubitRb
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib.pyplot as plt
import matplotlib


# todo: make sure to install: cirq, xarray, tqdm

matplotlib.use("TKAgg")

machine = QuAM.load()
config = machine.generate_config()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)
qc = machine.qubits["q4"]
qt = machine.qubits["q5"]


##############################
##  Two-qubit RB functions  ##
##############################
# single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a}
# that can reach any point on the Bloch sphere (starting from arbitrary points)
def bake_phased_xz(baker: Baking, q, x, z, a):
    if q == 1:
        element = qc.xy.name
    else:
        element = qt.xy.name

    baker.frame_rotation_2pi(a / 2, element)
    baker.play("x180", element, amp=x)
    baker.frame_rotation_2pi(-(a + z) / 2, element)


# single qubit phase corrections in units of 2pi applied after the CZ gate
qubit1_frame_update = 0.23  # example values, should be taken from QPU parameters
qubit2_frame_update = 0.12  # example values, should be taken from QPU parameters


# defines the CZ gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> -|11>
def bake_cz(baker: Baking, q1, q2):
    qc_xy_element = qc.xy.name
    qt_xy_element = qt.xy.name
    qc_z_element = qc.z.name

    baker.play("cz", qc_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit1_frame_update, qc_xy_element)
    baker.frame_rotation_2pi(qubit2_frame_update, qt_xy_element)
    baker.align()


def prep():
    wait(machine.thermalization_time)
    align()


def meas():
    threshold1 = 0.0  # threshold for state discrimination 0 <-> 1 using the I quadrature
    threshold2 = 0.0  # threshold for state discrimination 0 <-> 1 using the I quadrature
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len([qc, qt]))
    state1 = declare(bool)
    state2 = declare(bool)

    multiplexed_readout([qc, qt], I, I_st, Q, Q_st)

    assign(state1, I[0] > threshold1)  # assume that all information is in I
    assign(state2, I[1] > threshold2)  # assume that all information is in I

    return state1, state2


##############################
##  Two-qubit RB execution  ##
##############################
# create RB experiment from configuration and defined functions
rb = TwoQubitRb(
    config=config,
    single_qubit_gate_generator=bake_phased_xz,
    two_qubit_gate_generators={"CZ": bake_cz},  # can also provide e.g. "CNOT": bake_cnot
    prep_func=prep,
    measure_func=meas,
    interleaving_gate=None,
    verify_generation=False,
)

qmm = machine.connect()
res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=5, num_shots_per_circuit=100)
# circuit_depths ~ how many consecutive Clifford gates within one executed circuit
# (https://qiskit.org/documentation/apidoc/circuit.html)
# num_circuits_per_depth ~ how many random circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

data = {}

res.plot_hist()
plt.show()

res.plot_decay()
plt.show()

res.plot_fidelity()
plt.show()

A, alpha, B = res.fit_exponential()
fidelity = res.get_fidelity()
data["amplitude"] = A
data["decay_rate"] = alpha
data["mixed_state_probability"] = B
data["fidelity"] = fidelity
data["figure"] = plt.gcf()

node_save(machine, "two_qubit_randomized_benchmarking", data, additional_files=True)

# verify/save the random sequences created during the experiment
rb.save_sequences_to_file("sequences.txt")  # saves the gates used in each random sequence
# rb.save_command_mapping_to_file('commands.txt')  # saves mapping from "command id" to sequence
# rb.print_sequences()
# rb.print_command_mapping()
# rb.verify_sequences() # simulates random sequences to ensure they recover to ground state. takes a while...
