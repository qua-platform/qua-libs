from qiskit_aer.noise import depolarizing_error, NoiseModel
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap

num_qubits = 5
cm = CouplingMap.from_line(num_qubits, False)
error1q = 0.01  # single qubit gate error rate
error2q = 0.05  # two qubit gate error rate
depol_error1q = depolarizing_error(error1q, 1)
depol_error2q = depolarizing_error(error2q, 2)
sq_gate_set = ["h", "t", "sx", "ry", "sw"]
noise_model = NoiseModel(basis_gates=sq_gate_set)
if num_qubits == 2:
    noise_model.add_all_qubit_quantum_error(depol_error2q, ["cz", "cx"])
noise_model.add_all_qubit_quantum_error(depol_error1q, sq_gate_set)
# noise_model.add_all_qubit_quantum_error(depol_error1q, [ 'rx', 'sw', 'ry', 't'])
backend = AerSimulator(
    coupling_map=cm,
    noise_model=noise_model,
    method="density_matrix",
)
