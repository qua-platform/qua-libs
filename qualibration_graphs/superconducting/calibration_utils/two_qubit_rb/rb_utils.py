from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Literal, Set, Tuple, Union

from matplotlib import pyplot as plt
from qualibration_libs.core import BatchableList
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import Clifford
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator
from more_itertools import flatten


from scipy.optimize import curve_fit
import xarray

logger = logging.getLogger(__name__)

EPS = 1e-8


def get_multiplexed_pair_batches(
    machine: Any,
    parameters: Any,
    qubit_pairs: Union[List[str], List[Any]],
) -> BatchableList:
    """
    Group qubit pairs into batches: no shared qubits; no nearest-neighbor conflicts (grid + CZ spectators).
    If ``parameters.multiplexed`` is False, each pair is its own batch.
    """
    if machine is None:
        raise AttributeError("Machine is not set. Cannot collect qubit pairs.")

    pair_objects: List[Any] = []

    if len(qubit_pairs) > 0:
        if isinstance(qubit_pairs[0], str):
            for pair_name in qubit_pairs:
                if pair_name not in machine.qubit_pairs:
                    logger.warning("Pair '%s' not found in machine.qubit_pairs, skipping", pair_name)
                    continue
                pair_objects.append(machine.qubit_pairs[pair_name])
        else:
            pair_objects = list(qubit_pairs)
            for pair_obj in pair_objects:
                if not (
                    hasattr(pair_obj, "id")
                    or hasattr(pair_obj, "name")
                    or (hasattr(pair_obj, "qubit_control") and hasattr(pair_obj, "qubit_target"))
                ):
                    raise ValueError(f"Cannot determine pair identity for object {pair_obj}")

    is_multiplexed = hasattr(parameters, "multiplexed") and parameters.multiplexed

    if not is_multiplexed:
        batch_groups = [[i] for i in range(len(pair_objects))]
    else:

        def parse_grid_location(location_str: str) -> Tuple[int, int]:
            x, y = map(int, location_str.split(","))
            return (x, y)

        def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        qubit_grid_locations: Dict[str, Tuple[int, int]] = {}
        for qubit_name, qubit in machine.qubits.items():
            if hasattr(qubit, "grid_location") and qubit.grid_location:
                try:
                    qubit_grid_locations[qubit_name] = parse_grid_location(qubit.grid_location)
                except (ValueError, AttributeError):
                    pass

        nearest_neighbor_pairs: Set[Tuple[str, str]] = set()
        qubit_names = list(qubit_grid_locations.keys())
        for i in range(len(qubit_names)):
            q1 = qubit_names[i]
            p1 = qubit_grid_locations[q1]
            for j in range(i + 1, len(qubit_names)):
                q2 = qubit_names[j]
                p2 = qubit_grid_locations[q2]
                if manhattan_distance(p1, p2) == 1:
                    nearest_neighbor_pairs.add((q1, q2))
                    nearest_neighbor_pairs.add((q2, q1))

        def are_nearest_neighbors(q1: str, q2: str) -> bool:
            return (q1, q2) in nearest_neighbor_pairs

        batch_groups: List[List[int]] = []
        pair_qubits: Dict[int, Set[str]] = {}
        pair_spectator_count: Dict[int, int] = {}

        for idx, pair_obj in enumerate(pair_objects):
            qubits_set = {pair_obj.qubit_control.name, pair_obj.qubit_target.name}
            spectator_count = 0
            if hasattr(pair_obj, "macros") and "cz" in pair_obj.macros:
                cz_macro = pair_obj.macros["cz"]
                if hasattr(cz_macro, "spectator_qubits") and cz_macro.spectator_qubits:
                    for spectator_qubit_name in cz_macro.spectator_qubits.keys():
                        qubits_set.add(spectator_qubit_name)
                        spectator_count += 1
            pair_qubits[idx] = qubits_set
            pair_spectator_count[idx] = spectator_count

        sorted_pair_indices = sorted(
            pair_qubits.keys(),
            key=lambda idx: pair_spectator_count[idx],
            reverse=True,
        )

        batch_qubits: List[Set[str]] = []

        for pair_idx in sorted_pair_indices:
            qubits_in_pair = pair_qubits[pair_idx]
            batch_found = False
            for i, batch_qubit_set in enumerate(batch_qubits):
                if not qubits_in_pair.isdisjoint(batch_qubit_set):
                    continue
                conflict_with_nn = False
                for qubit_in_pair in qubits_in_pair:
                    for existing_qubit in batch_qubit_set:
                        if are_nearest_neighbors(qubit_in_pair, existing_qubit):
                            conflict_with_nn = True
                            break
                    if conflict_with_nn:
                        break
                if conflict_with_nn:
                    continue
                batch_groups[i].append(pair_idx)
                batch_qubits[i].update(qubits_in_pair)
                batch_found = True
                break
            if not batch_found:
                batch_groups.append([pair_idx])
                batch_qubits.append(qubits_in_pair.copy())

    return BatchableList(pair_objects, batch_groups)

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
            
            if length > 0:
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


def validate_multiplexed_batches(qubit_pairs, multiplexed: bool):
    """
    If ``multiplexed`` is True, there must be exactly one batch (all pairs run together).
    If ``multiplexed`` is False, each batch must contain only one pair.
    """
    batches = list(qubit_pairs.batch())

    if multiplexed:
        if len(batches) > 1:
            raise ValueError(
                f"Unsupported configuration: Found {len(batches)} multiplexed batches, "
                "but only one batch is supported. Please run batches separately."
            )
    else:
        for batch_idx, batch in enumerate(batches, 1):
            if len(batch) > 1:
                pair_names = [qp.id if hasattr(qp, "id") else str(qp) for qp in batch.values()]
                raise ValueError(
                    f"Unsupported configuration: Batch {batch_idx} contains {len(batch)} pairs "
                    f"({', '.join(pair_names)}), but multiplexed=False. "
                    "Either set multiplexed=True or run pairs separately."
                )

   