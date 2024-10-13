import dataclasses
import os
import pathlib
import pickle
import random
from typing import Set, List

import cirq
import numpy as np

from .simple_tableau import SimpleTableau

q1, q2 = cirq.LineQubit.range(1, 3)

C1_reduced = [
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0),
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=-0.5, z_exponent=0),
    cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=1),
    cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5),
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5),
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0.5),
]

S1 = [
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0),
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5),
    cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5),
]

pauli = [
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0),  # I
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=1.0, z_exponent=0),  # X
    cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1.0, z_exponent=0),  # Y
    cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1.0),  # Z
]

pauli_phase = [
    [0, 0],  # I
    [0, 1],  # X
    [1, 1],  # Y
    [1, 0],  # Z
]

native_2_qubit_gates = {
    "sqr_iSWAP": {
        "CNOT": [
            cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=-1)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(q2),
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=1, z_exponent=0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0)(q2),
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0.5)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q2),
        ],  # compilation of CNOT in terms of phased XZ and sqiSWAP
        "iSWAP": [
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.ISWAP(q1, q2) ** 0.5,
        ],  # compilation of iSWAP in terms of phased XZ and sqiSWAP
        "SWAP": [
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q2),
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q2),
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q2),
            cirq.ISWAP(q1, q2) ** 0.5,
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(q2),
        ],  # compilation of SWAP in terms of phased XZ and sqiSWAP
    },
    # TODO: add more implementations
    "CNOT": {
        "CNOT": [cirq.CNOT(q1, q2)],
        "iSWAP": [
            cirq.PhasedXZGate(axis_phase_exponent=-1.0, x_exponent=0.5, z_exponent=-0.5)(q1),  # S + H -> PhasedXZ
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.5)(q2),  # S
            cirq.CNOT(q1, q2),
            cirq.CNOT(q2, q1),
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.0)(q1),  # I
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),  # H -> PhasedXZ
        ],
        "SWAP": [cirq.CNOT(q2, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q1)],
    },
    "CZ": {
        "CNOT": [
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.0)(q1),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.0)(q1),
        ],
        "iSWAP": [
            cirq.PhasedXZGate(axis_phase_exponent=-1.0, x_exponent=0.5, z_exponent=-0.5)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-1.0, x_exponent=0.5, z_exponent=-0.5)(q2),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
        ],
        "SWAP": [
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.0)(q1),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q1),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.CZ(q1, q2),
            cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.0)(q2),
            cirq.PhasedXZGate(axis_phase_exponent=0.0, x_exponent=0.0, z_exponent=0.0)(q1),
        ],
    },
}


##### Conversion from unitary / cirq to tableau

I = np.matrix([[1, 0], [0, 1]])  # identity
X = np.matrix([[0, 1], [1, 0]])  # pi x
Y = np.matrix([[0, -1j], [1j, 0]])  # pi y
Z = np.matrix([[1, 0], [0, -1]])  # pi z

paulis = [I, X, Y, Z]
twoQBPaulis = [np.kron(i, j) for i in paulis for j in paulis]

symplecticTable = [
    [0, 0, 0, 0],  # II
    [0, 0, 1, 0],  # IX
    [0, 0, 1, 1],  # IY
    [0, 0, 0, 1],  # IZ
    [1, 0, 0, 0],  # XI
    [1, 0, 1, 0],  # XX
    [1, 0, 1, 1],  # XY
    [1, 0, 0, 1],  # XZ
    [1, 1, 0, 0],  # YI
    [1, 1, 1, 0],  # YX
    [1, 1, 1, 1],  # YY
    [1, 1, 0, 1],  # YZ
    [0, 1, 0, 0],  # ZI
    [0, 1, 1, 0],  # ZX
    [0, 1, 1, 1],  # ZY
    [0, 1, 0, 1],
]  # ZZ


def get_pauli_prod(m):
    # input: tensor product of two Pauli matrices
    # output: tableau column of m

    for i, p in enumerate(twoQBPaulis):
        prod = m @ p
        if np.trace(prod) > 3.9:
            return symplecticTable[i], 0
        if np.trace(prod) < -3.9:
            return symplecticTable[i], 1


def tableau_from_unitary(m):
    # Turns a two-qubit unitary into the full tableau representation
    # input: two-qubit unitary m
    # outputs: simple tableau
    s = np.zeros([4, 4])
    p = np.zeros(4)

    s[:, 0], p[0] = get_pauli_prod(m @ np.kron(X, I) @ m.H)
    s[:, 1], p[1] = get_pauli_prod(m @ np.kron(Z, I) @ m.H)
    s[:, 2], p[2] = get_pauli_prod(m @ np.kron(I, X) @ m.H)
    s[:, 3], p[3] = get_pauli_prod(m @ np.kron(I, Z) @ m.H)

    return SimpleTableau(s, p)


def tableau_from_cirq(gates: List[cirq.GateOperation]) -> SimpleTableau:
    return tableau_from_unitary(np.matrix(cirq.Circuit(gates).unitary()))


#########################################################


def combine_to_phased_x_z(first_gate: cirq.GateOperation, second_gate: cirq.GateOperation) -> cirq.GateOperation:
    unitary = cirq.Circuit([first_gate, second_gate]).unitary()
    if unitary.shape != (2, 2):
        raise RuntimeError("Cannot combine multi qubit gate to PhasedXZ")
    return cirq.PhasedXZGate.from_matrix(unitary)(first_gate.qubits[0])


@dataclasses.dataclass
class GateCommand:
    type: str
    q1: tuple
    q2: tuple

    def get_qubit_ops(self, q):
        if q == 0:
            return self.q1
        elif q == 1:
            return self.q2
        else:
            raise RuntimeError("q should be 0 or 1")


class _GateDatabase:
    def __init__(self):
        self._commands, self._tableaus, self._symplectic_range, self._pauli_range = self._gen_commands_and_tableaus()

    @staticmethod
    def _gen_commands_and_tableaus():
        compilation_path = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "symplectic_compilation_XZ.pkl"
        with open(compilation_path, "rb") as f:
            compilation = pickle.load(f)
        symplectics = compilation["symplectics"]
        phases = compilation["phases"]
        commands = compilation["commands"]

        rb_commands = []
        for command in commands:
            if command[0] == "C1's":
                rb_commands.append(GateCommand("C1", (command[1],), (command[2],)))
            elif command[0] == "CNOT's":
                rb_commands.append(GateCommand("CNOT", (command[1], command[3]), (command[2], command[4])))
            elif command[0] == "iSWAP's":
                rb_commands.append(GateCommand("iSWAP", (command[1], command[3]), (command[2], command[4])))
            elif command[0] == "SWAP's":
                rb_commands.append(GateCommand("SWAP", (command[1],), (command[2],)))

        # Generate Paulis:
        for i1 in range(len(pauli)):
            for i2 in range(len(pauli)):
                rb_commands.append(GateCommand("PAULI", (i1,), (i2,)))

        tableaus = []
        for i in range(len(symplectics)):
            tableaus.append(SimpleTableau(symplectics[i], phases[i]))

        # Generate Paulis:
        for i1 in range(len(pauli)):
            for i2 in range(len(pauli)):
                tableaus.append(SimpleTableau(np.eye(4), pauli_phase[i1] + pauli_phase[i2]))

        symplectic_range = (0, len(commands))
        pauli_range = (len(commands), len(rb_commands))
        return rb_commands, tableaus, symplectic_range, pauli_range

    @property
    def commands(self):
        return self._commands

    @property
    def tableaus(self):
        return self._tableaus

    def get_command(self, gate_id) -> GateCommand:
        return self._commands[gate_id]

    def get_tableau(self, gate_id) -> SimpleTableau:
        return self._tableaus[gate_id]

    def rand_symplectic(self):
        return random.randrange(*self._symplectic_range)

    def rand_pauli(self):
        return random.randrange(*self._pauli_range)

    def get_interleaving_gate(self):
        return self._pauli_range[1]

    def find_symplectic_gate_id_by_tableau_g(self, tableau: SimpleTableau):
        tableaus = self._tableaus[self._symplectic_range[0] : self._symplectic_range[1]]
        return next(i for i, x in enumerate(tableaus) if np.array_equal(x.g, tableau.g))

    def find_pauli_gate_id_by_tableau_alpha(self, tableau: SimpleTableau):
        tableaus = self._tableaus[self._pauli_range[0] : self._pauli_range[1]]
        return self._pauli_range[0] + next(i for i, x in enumerate(tableaus) if np.array_equal(x.alpha, tableau.alpha))


gate_db = _GateDatabase()


class GateGenerator:
    two_qubit_imp_priority = {  # TODO: verify this priority table
        "CNOT": ["CNOT", "CZ", "iSWAP", "sqr_iSWAP"],
        "iSWAP": ["iSWAP", "sqr_iSWAP", "CNOT", "CZ"],
        "SWAP": ["CNOT", "CZ", "iSWAP", "sqr_iSWAP"],
    }

    def __init__(self, native_two_qubit_gates: Set[str]):
        self._two_qubit_dict = self._generate_two_qubit_dict(native_two_qubit_gates)

    @staticmethod
    def _generate_two_qubit_dict(native_two_qubit_gates: Set[str]) -> dict:
        two_qubit_dict = {}
        for k, v in GateGenerator.two_qubit_imp_priority.items():
            available_imp = [x for x in v if x in native_two_qubit_gates]
            if len(available_imp) == 0 or available_imp[0] not in native_2_qubit_gates.keys():
                raise RuntimeError(f"Cannot implement gate '{k}' with provided native two qubit gates")
            two_qubit_dict[k] = available_imp[0]
        return two_qubit_dict

    @staticmethod
    def _reduce_gate(gate: List[cirq.GateOperation]):
        qubit_ops = {1: None, 2: None}
        output = []

        def append_qubit_ops():
            for q in [1, 2]:
                if qubit_ops[q] is not None:
                    output.append(qubit_ops[q])
                qubit_ops[q] = None

        for op in gate:
            if len(op.qubits) == 1:
                prev_op = qubit_ops[op.qubits[0].x]
                qubit_ops[op.qubits[0].x] = op if prev_op is None else combine_to_phased_x_z(prev_op, op)
            else:
                append_qubit_ops()
                output.append(op)
        append_qubit_ops()
        return output

    def generate(self, cmd_id):
        gate = []
        command = gate_db.get_command(cmd_id)
        two_qubit_imp = self._two_qubit_dict[command.type] if command.type in self._two_qubit_dict else None
        if command.type == "C1":
            gate.append(C1_reduced[command.q1[0]](q1))
            gate.append(C1_reduced[command.q2[0]](q2))
        elif command.type == "CNOT":
            gate.append(C1_reduced[command.q1[0]](q1))
            gate.append(C1_reduced[command.q2[0]](q2))
            gate.extend(native_2_qubit_gates[two_qubit_imp]["CNOT"])
            gate.append(S1[command.q1[1]](q1))
            gate.append(S1[command.q2[1]](q2))
        elif command.type == "iSWAP":
            gate.append(C1_reduced[command.q1[0]](q1))
            gate.append(C1_reduced[command.q2[0]](q2))
            gate.extend(native_2_qubit_gates[two_qubit_imp]["iSWAP"])
            gate.append(S1[command.q1[1]](q1))
            gate.append(S1[command.q2[1]](q2))
        elif command.type == "SWAP":
            gate.append(C1_reduced[command.q1[0]](q1))
            gate.append(C1_reduced[command.q2[0]](q2))
            gate.extend(native_2_qubit_gates[two_qubit_imp]["SWAP"])
        elif command.type == "PAULI":
            gate.append(pauli[command.q1[0]](q1))
            gate.append(pauli[command.q2[0]](q2))
        else:
            raise RuntimeError(f"unknown command {command.type}")
        return self._reduce_gate(gate)
