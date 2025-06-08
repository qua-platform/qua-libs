import copy
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import Clifford
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator
from more_itertools import flatten


from scipy.optimize import curve_fit
import xarray

EPS = 1e-8

def rb_decay_curve(x, A, alpha, B):
    """
    Exponential decay model for RB fidelity.

    Args:
        x (array-like): Circuit depths.
        A (float): Amplitude of the decay.
        alpha (float): Decay constant.
        B (float): Offset of the curve.

    Returns:
        np.ndarray: Calculated decay curve.
    """
    return A * alpha**x + B


class RBBase:
    
    def __init__(self, circuit_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None):
        
        self.num_qubits = num_qubits
        self.circuit_lengths = circuit_lengths
        self.num_circuits_per_length = num_circuits_per_length
        
        self.basis_gates = basis_gates
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.rolling_seed = copy.deepcopy(self.seed)
        self.reduce_to_1q_cliffords = reduce_to_1q_cliffords
    
    def generate_circuits_and_transpile(self, interleaved: bool = False):
        
        self.circuits = self.generate_circuits(interleaved)
        self.transpiled_circuits = {l : self.transpile_per_clifford(circuits) for l, circuits in self.circuits.items()}
        
    def generate_circuits_per_length(self, length: int, interleaved: bool = False) -> list[QuantumCircuit]:
        
        circuits = []
        
        for _ in range(self.num_circuits_per_length):
            qc = QuantumCircuit(self.num_qubits)
            clifford_product = Clifford(qc)  # Identity Clifford
            
            # Apply random Clifford gates
            for _ in range(length):
                if self.reduce_to_1q_cliffords and self.num_qubits == 2:
                    qc_temp = QuantumCircuit(2)
                    cliff = random_clifford(1, self.rolling_seed)
                    self.rolling_seed += 1
                    qc_temp.append(cliff, (0,))
                    cliff = random_clifford(1, self.rolling_seed)
                    self.rolling_seed += 1
                    qc_temp.append(cliff, (1,))
                    cliff = Clifford(qc_temp)
                else:
                    cliff = random_clifford(self.num_qubits, self.rolling_seed)
                    self.rolling_seed += 1
                
                qc.append(cliff, range(self.num_qubits))
                
                if interleaved:
                    if not hasattr(self, 'target_gate_instruction'):
                        raise AttributeError("The attribute 'target_gate_instruction' is not defined in the class.")
                    
                    qc.append(self.target_gate_instruction, range(self.num_qubits))
                    cliff = Clifford(self.target_gate_instruction) @ cliff
                
                clifford_product = cliff @ clifford_product  # Update the total Clifford
            
            # Append the inverse Clifford
            inverse_clifford = clifford_product.adjoint()
            qc.append(inverse_clifford, range(self.num_qubits))
            
            # # Verify that the quantum circuit is an identity operator up to a phase
            # unitary = Operator(qc).data
            # identity = np.eye(unitary.shape[0])
            # # Normalize the unitary to remove global phase
            # unitary_normalized = unitary / np.linalg.det(unitary)**(1/unitary.shape[0])
            # assert np.allclose(unitary_normalized, identity, atol=1e-8), "Circuit is not an identity operator up to a phase."
            
            circuits.append(qc)
        
        return circuits
    
    def generate_circuits(self, interleaved: bool = False) -> dict[int, list[QuantumCircuit]]:
        
        circuits = {}
        for len in self.circuit_lengths:
            circuits_per_len = self.generate_circuits_per_length(len, interleaved)
            circuits[len] = circuits_per_len
        
        return circuits
    
    def transpile_per_clifford(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
         
        transpiled_circuits = []
        
        for qc in circuits:
            transp_circ = QuantumCircuit(self.num_qubits)
            for instruction in qc:
                qc_per_inst = QuantumCircuit(len(instruction.qubits))
                qc_per_inst.append(instruction)
                
                if isinstance(instruction.operation, Clifford):
                    # if optimization level is > 1 one might get fractional angles
                    transpiled_gate = transpile(qc_per_inst, basis_gates=self.basis_gates, optimization_level=1)
                else:
                    transpiled_gate = qc_per_inst.copy()
                
                transp_circ = transp_circ.compose(transpiled_gate, front=False)
            
            transpiled_circuits.append(transp_circ)
        
        return transpiled_circuits
    
    def count_num_gates(self) -> int:
        return sum([len(qc) for qc in flatten(self.transpiled_circuits.values())])
    
    def plot_with_fidelity(self, data: xarray, num_averages: int):
        """
        Plots the RB fidelity as a function of circuit depth, including a fit to an exponential decay model.
        The fitted curve is overlaid with the raw data points.
        """
        A, alpha, B = self.fit_exponential(data, num_averages)
        fidelity = self.get_fidelity(alpha)

        plt.figure()
        plt.plot(self.circuit_lengths, self.get_decay_curve(data, num_averages), "o", label="Data")
        plt.plot(
            self.circuit_lengths,
            rb_decay_curve(np.array(self.circuit_lengths), A, alpha, B),
            "-",
            label=f"Fidelity={fidelity * 100:.3f}%\nalpha={alpha:.4f}",
        )
        plt.xlabel("Circuit Depth")
        plt.ylabel("Fidelity")
        plt.title("2Q Randomized Benchmarking Fidelity")
        plt.legend()
        plt.show()

    def fit_exponential(self, data: xarray, num_averages: int):
        """
        Fits the decay curve of the RB data to an exponential model.

        Returns:
            tuple: Fitted parameters (A, alpha, B) where:
                - A is the amplitude.
                - alpha is the decay constant.
                - B is the offset.
        """
        decay_curve = self.get_decay_curve(data, num_averages)

        popt, _ = curve_fit(rb_decay_curve, self.circuit_lengths, decay_curve, p0=[0.75, -0.1, 0.25], maxfev=10000)
        A, alpha, B = popt

        return A, alpha, B

    def get_fidelity(self, alpha):
        """
        Calculates the average fidelity per Clifford based on the decay constant.

        Args:
            alpha (float): Decay constant from the exponential fit.

        Returns:
            float: Estimated average fidelity per Clifford.
        """
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self, data: xarray, num_averages: int):
        """
        Calculates the decay curve from the RB data.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        return (data.state == 0).sum(("qubit", "num_circuits_per_length", "N")) / (self.num_circuits_per_length * num_averages)


class StandardRB(RBBase):
    
    def __init__(self, amplification_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None):
        
        super().__init__(amplification_lengths, num_circuits_per_length, basis_gates, num_qubits, reduce_to_1q_cliffords, seed)
        
        self.generate_circuits_and_transpile()

class InterleavedRB(RBBase):
    
    def __init__(self, target_gate: Literal['cz', 'idle_2q'], amplification_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None):
        
        self.target_gate = target_gate
        self.target_gate_instruction = self.target_gate_to_instruction()
        
        super().__init__(amplification_lengths, num_circuits_per_length, basis_gates, num_qubits, reduce_to_1q_cliffords, seed)
        
        self.generate_circuits_and_transpile(interleaved=True)
    
    def target_gate_to_instruction(self) -> Instruction:
        
        qc = QuantumCircuit(2)
        if self.target_gate == 'cz':
            qc.cz(0, 1)
        elif self.target_gate == 'idle_2q':
            qc.id((0, 1))
        else:
            raise ValueError(f"Target gate {self.target_gate} not supported")
        
        instruction = qc.to_instruction()
        instruction.name = self.target_gate
        return instruction

def layerize_quantum_circuit(qc: QuantumCircuit) -> QuantumCircuit:

    dag = circuit_to_dag(qc)
    layered_qc = QuantumCircuit(qc.num_qubits)
    for layer in dag.layers():
        layer_as_circuit = dag_to_circuit(layer['graph'])
        layer_as_circuit.barrier()
        layered_qc = layered_qc.compose(layer_as_circuit)

    return layered_qc

def collect_gates_as_2_qubit_layers(qc: QuantumCircuit) -> list[str]:
    """
    Iterates over a qiskit quantum circuit and collects the gates as strings.
    The circuit is separated by barriers. Between each barrier, the function collects the gates as strings.
    The name of the gate as string should be like <name>_<qubit>. Between gates place a dash.
    
    Args:
        qc (QuantumCircuit): The quantum circuit to iterate over.
    
    Returns:
        list[str]: A list of strings representing the gates in the circuit.
    """
    gate_strings = []
    current_gates = []

    for instruction in qc:
        if instruction.name == 'barrier':
            if current_gates:
                gate_strings.append('-'.join(current_gates))
                current_gates = []
        else:
            for qubit in instruction.qubits:
                gate_str = f"{instruction.name}_{qubit.index}"
                current_gates.append(gate_str)
    
    if current_gates:
        gate_strings.append('-'.join(current_gates))
    
    return gate_strings


def quantum_circuit_to_integer_and_angle_list(qc: QuantumCircuit, gate_map: dict, ignore_barriers: bool = True) -> tuple[list[int], list[float]]:

    gate_interger_list = []
    rz_angle_count_q0 = []
    rz_angle_count_q1 = []
    qubit_index_list = [] # need to make this more robust

    layered_qc = layerize_quantum_circuit(qc)

    ang_q0 = 0
    ang_q1 = 0
    
    for instruction in layered_qc:
        
        qubit_index = instruction.qubits[0]._index # TODO : make this more robust
        
        if instruction.name == 'rz':
            # only follows the accumulation of rz angles. does not add a to gate list
            if qubit_index == 0:
                ang_q0 += instruction.params[0]
            elif qubit_index == 1:
                ang_q1 += instruction.params[0]
            else:
                raise ValueError("Only 2 qubits are supported")
        
        else:
            # add a gate
            if instruction.name in ['rx', 'ry']:
                axis = instruction.name[1]
                theta = instruction.params[0]
                if (theta/np.pi - 0.5) < EPS:
                    gate_int = gate_map[f's{axis}']
                elif (theta/np.pi - (-0.5)) < EPS:
                    gate_int = gate_map[f'{axis}270']
                elif (theta/np.pi - 1) < EPS or (theta/np.pi - (-1)) < EPS:
                    gate_int = gate_map[f'{axis}']
                else:
                    raise ValueError(f"Only 90, -90, 180, -180 rotations are supported not angle={theta}")
            
            elif instruction.name == 'barrier' and ignore_barriers:
                continue
            
            else:
                gate_int = gate_map[instruction.name]
            
            gate_interger_list.append(gate_int)
            qubit_index_list.append(qubit_index)
            rz_angle_count_q0.append(ang_q0)
            rz_angle_count_q1.append(ang_q1)
            
            ang_q0 = 0  
            ang_q1 = 0
            

    # measurement
    gate_interger_list.append(gate_map['measure'])
    rz_angle_count_q0.append(0)
    rz_angle_count_q1.append(0)
    qubit_index_list.append(0)

    return gate_interger_list, qubit_index_list, rz_angle_count_q0, rz_angle_count_q1
    
   