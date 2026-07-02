"""Randomized benchmarking circuit generation utilities.

This module provides classes for generating standard and interleaved
randomized benchmarking circuits.
"""

# pylint: disable=duplicate-code,use-implicit-booleaness-not-comparison-to-zero

import copy
from typing import Literal

import numpy as np
from more_itertools import flatten
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import *
from qiskit.quantum_info import Clifford, Operator, random_clifford
from tqdm.auto import tqdm


class RBBase:  # pylint: disable=too-many-instance-attributes
    """Base class for randomized benchmarking circuit generation."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        circuit_lengths: list[int],
        num_circuits_per_length: int,
        basis_gates: list[str] = ["cz", "rz", "sx", "x"],
        num_qubits: int = 2,
        reduce_to_1q_cliffords: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the RB base class.

        Args:
            circuit_lengths: List of circuit depths (number of Cliffords) to generate.
            num_circuits_per_length: Number of random circuits to generate per depth.
            basis_gates: List of basis gates for transpilation.
            num_qubits: Number of qubits in the circuits.
            reduce_to_1q_cliffords: Whether to use single-qubit Cliffords for 2-qubit RB.
            seed: Random seed for circuit generation.
            show_progress: Whether to display tqdm progress bars during circuit
                generation and transpilation. Default True.
        """

        self.num_qubits = num_qubits
        self.circuit_lengths = circuit_lengths
        self.num_circuits_per_length = num_circuits_per_length

        self.basis_gates = basis_gates
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.rolling_seed = copy.deepcopy(self.seed)
        self.reduce_to_1q_cliffords = reduce_to_1q_cliffords
        self.show_progress = show_progress

    def generate_circuits_and_transpile(self, interleaved: bool = False):
        """Generate circuits and transpile them to the basis gate set."""
        self.circuits = self.generate_circuits(interleaved)
        items = self.circuits.items()
        if self.show_progress:
            items = tqdm(items, total=len(self.circuits), desc="Transpiling RB circuits", unit="depth")
        self.transpiled_circuits = {l: self.transpile_per_clifford(circuits) for l, circuits in items}

    def generate_circuits_per_length(
        self,
        length: int,
        interleaved: bool = False,
        progress: "tqdm | None" = None,
    ) -> list[QuantumCircuit]:
        """
        Generate random Clifford circuits for a given length.

        Args:
            length: Number of Cliffords in the circuit.
            interleaved: Whether to interleave a target gate between Cliffords.
            progress: Optional tqdm progress bar. If provided, advances by 1
                per Clifford appended.

        Returns:
            List of generated quantum circuits.
        """

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
                    if not hasattr(self, "target_gate_instruction"):
                        raise AttributeError("The attribute 'target_gate_instruction' is not defined in the class.")

                    qc.append(self.target_gate_instruction, range(self.num_qubits))  # pylint: disable=no-member
                    cliff = Clifford(self.target_gate_instruction) @ cliff  # pylint: disable=no-member

                clifford_product = cliff @ clifford_product  # Update the total Clifford
                if progress is not None:
                    progress.update(1)

            # Append the inverse Clifford
            inverse_clifford = clifford_product.adjoint()
            qc.append(inverse_clifford, range(self.num_qubits))

            circuits.append(qc)

        return circuits

    def generate_circuits(self, interleaved: bool = False) -> dict[int, list[QuantumCircuit]]:
        """
        Generate circuits for all specified lengths.

        Args:
            interleaved: Whether to interleave a target gate.

        Returns:
            Dictionary mapping circuit lengths to lists of circuits.
        """
        # Use total Cliffords across all (depth, circuit_index) pairs as the
        # progress unit so the bar advances smoothly even when individual
        # depths take very different amounts of time.
        total_cliffords = self.num_circuits_per_length * sum(self.circuit_lengths)
        pbar = (
            tqdm(total=total_cliffords, desc="Generating RB Cliffords", unit="cliff")
            if self.show_progress
            else None
        )
        try:
            circuits = {}
            for length in self.circuit_lengths:
                circuits_per_len = self.generate_circuits_per_length(length, interleaved, progress=pbar)
                circuits[length] = circuits_per_len
            return circuits
        finally:
            if pbar is not None:
                pbar.close()

    def transpile_per_clifford(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
        """
        Transpile circuits to the basis gate set.

        Args:
            circuits: List of circuits to transpile.

        Returns:
            List of transpiled circuits.
        """

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
        """Count the total number of gates across all transpiled circuits."""
        return sum(len(qc) for qc in flatten(self.transpiled_circuits.values()))


class StandardRB(RBBase):
    """Class for generating standard randomized benchmarking circuits."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        amplification_lengths: list[int],
        num_circuits_per_length: int,
        basis_gates: list[str] = ["cz", "rz", "sx", "x"],
        num_qubits: int = 2,
        reduce_to_1q_cliffords: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize standard RB circuit generator.

        Args:
            amplification_lengths: List of circuit depths to generate.
            num_circuits_per_length: Number of random circuits per depth.
            basis_gates: List of basis gates for transpilation.
            num_qubits: Number of qubits in the circuits.
            reduce_to_1q_cliffords: Whether to use single-qubit Cliffords.
            seed: Random seed for circuit generation.
            show_progress: Whether to display tqdm progress bars. Default True.
        """

        super().__init__(
            amplification_lengths,
            num_circuits_per_length,
            basis_gates,
            num_qubits,
            reduce_to_1q_cliffords,
            seed,
            show_progress,
        )

        self.generate_circuits_and_transpile()


class InterleavedRB(RBBase):
    """Class for generating interleaved randomized benchmarking circuits."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        target_gate: Literal["cz", "idle_2q"],
        amplification_lengths: list[int],
        num_circuits_per_length: int,
        basis_gates: list[str] = ["cz", "rz", "sx", "x"],
        num_qubits: int = 2,
        reduce_to_1q_cliffords: bool = False,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize interleaved RB circuit generator.

        Args:
            target_gate: The gate to interleave between Cliffords.
            amplification_lengths: List of circuit depths to generate.
            num_circuits_per_length: Number of random circuits per depth.
            basis_gates: List of basis gates for transpilation.
            num_qubits: Number of qubits in the circuits.
            reduce_to_1q_cliffords: Whether to use single-qubit Cliffords.
            seed: Random seed for circuit generation.
            show_progress: Whether to display tqdm progress bars. Default True.
        """

        self.target_gate = target_gate
        self.target_gate_instruction = self.target_gate_to_instruction()

        super().__init__(
            amplification_lengths,
            num_circuits_per_length,
            basis_gates,
            num_qubits,
            reduce_to_1q_cliffords,
            seed,
            show_progress,
        )

        self.generate_circuits_and_transpile(interleaved=True)

    def target_gate_to_instruction(self) -> Instruction:
        """
        Convert the target gate name to a Qiskit instruction.

        Returns:
            A Qiskit instruction representing the target gate.
        """
        qc = QuantumCircuit(2)
        if self.target_gate == "cz":
            qc.cz(0, 1)
        elif self.target_gate == "idle_2q":
            qc.id((0, 1))
        else:
            raise ValueError(f"Target gate {self.target_gate} not supported")

        instruction = qc.to_instruction()
        instruction.name = self.target_gate
        return instruction
