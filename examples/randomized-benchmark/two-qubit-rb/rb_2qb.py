from typing import Optional, List

from qiskit import QuantumCircuit, Aer, execute
import qiskit.circuit.library as glib
import numpy as np
from configuration import config
from qualang_tools.bakery.bakery import baking
from qm.qua import *
# this builds up on https://arxiv.org/pdf/1402.4848
# build up sequence

_generate_table = False
_test_2_design = False  # careful, this takes about 20 minutes
size_c2 = 11520

_c1_ops = [
    ('I',),
    ('X',),
    ('Y',),
    ('Y', 'X'),

    ('X/2', 'Y/2'),
    ('X/2', '-Y/2'),
    ('-X/2', 'Y/2'),
    ('-X/2', '-Y/2'),
    ('Y/2', 'X/2'),
    ('Y/2', '-X/2'),
    ('-Y/2', 'X/2'),
    ('-Y/2', '-X/2'),

    ('X/2',),
    ('-X/2',),
    ('Y/2',),
    ('-Y/2',),
    ('-X/2', 'Y/2', 'X/2'),
    ('-X/2', '-Y/2', 'X/2'),

    ('X', 'Y/2'),
    ('X', '-Y/2'),
    ('Y', 'X/2'),
    ('Y', '-X/2'),
    ('X/2', 'Y/2', 'X/2'),
    ('-X/2', 'Y/2', '-X/2'),

]

_s1_ops = [
    ('I',),
    ('Y/2', 'X/2'),
    ('-X/2', '-Y/2'),
]

_s1x2_ops = [
    ('X/2',),
    ('X/2', 'Y/2', 'X/2'),
    ('-Y/2',)
]

_s1y2_ops = [
    ('Y/2',),
    ('Y', 'X/2'),
    ('-X/2', '-Y/2', 'X/2')
]


class RBTwoQubits:
    def __init__(self, config: dict, d_max: int, K: int, gate_macros: dict, *quantum_elements: str):
        """
        Class to retrieve easily baked RB sequences and their inverse operations
        :param config Configuration file
        :param d_max Maximum length of desired RB sequence
        :param K Number of RB sequences
        :param quantum_elements Set of quantum elements allowing to perform 2 qb RB (usually in format q1, q2, coupler)
        """
        for qe in quantum_elements:
            if qe not in config["elements"]:
                raise KeyError(f"Quantum element {qe} is not in the config")

        self.sequences = [TwoQbRBSequence(config, d_max, gate_macros, *quantum_elements) for _ in range(K)]
        self.inverse_ops = [seq.revert_ops for seq in self.sequences]
        self.duration_trackers = [seq.duration_tracker for seq in self.sequences]
        self.baked_sequences = [seq.sequence for seq in self.sequences]


class TwoQbRBSequence:
    def __init__(self, config: dict, d_max: int, gate_macros: dict, *quantum_elements: str):
        self.d_max = d_max
        self.config = config
        self.quantum_elements = quantum_elements
        self.gate_macros = gate_macros
        self.state_tracker = [
            0
        ] * d_max  # Keeps track of all transformations done on qubit state
        self.state_init = 0
        self.revert_ops = [
            0
        ] * d_max  # Keeps track of inverse op index associated to each sequence
        self.duration_tracker = [0] * d_max  # Keeps track of each Clifford's duration
        # self.baked_cliffords = self.generate_cliffords()  # List of baking objects for running Cliffords
        self.operations_list = [None] * d_max
        self.inverse_op_string = [""] * d_max
        self.sequence = self.generate_RB_sequence()  # Store the RB sequence


def index_to_clifford(index):
    if index < 576:
        # single qubit class
        q0c1, q1c1 = np.unravel_index(index, (24, 24))
        gate = [(_c1_ops[q0c1],), (_c1_ops[q1c1],)]
    elif index < 576 + 5184:
        # CNOT class
        index -= 576
        q0c1, q1c1, q0s1, q1s1y2 = np.unravel_index(index, (24, 24, 3, 3))
        gate = [(_c1_ops[q0c1], 'cz', _s1_ops[q0s1]),
                (_c1_ops[q1c1], 'cz', _s1y2_ops[q1s1y2])]
    elif index < 576 + 2 * 5184:
        # iSWAP class
        index -= 576 + 5184
        q0c1, q1c1, q0s1y2, q1s1x2 = np.unravel_index(index, (24, 24, 3, 3))
        gate = [(_c1_ops[q0c1], 'cz', ('Y/2',), 'cz', _s1y2_ops[q0s1y2]),
                (_c1_ops[q1c1], 'cz', ('-X/2',), 'cz', _s1x2_ops[q1s1x2])]
    else:
        # swap class
        index -= 576 + 2 * 5184
        q0c1, q1c1 = np.unravel_index(index, (24, 24))
        gate = [(_c1_ops[q0c1], 'cz', ('-Y/2',), 'cz', ('Y/2',), 'cz', ('I',)),
                (_c1_ops[q1c1], 'cz', ('Y/2',), 'cz', ('-Y/2',), 'cz', ('Y/2',))]
    return gate


_single_gate_to_qiskit = {
    'I': glib.IGate(),
    'X': glib.XGate(),
    'Y': glib.YGate(),
    'X/2': glib.RXGate(np.pi / 2),
    'Y/2': glib.RYGate(np.pi / 2),
    '-X/2': glib.RXGate(-np.pi / 2),
    '-Y/2': glib.RYGate(-np.pi / 2),
}


def _clifford_to_qiskit_circ(clifford):
    qc = QuantumCircuit(2)
    for q0g, q1g in zip(*clifford):

        if q0g == 'cz':
            qc.append(glib.CZGate(), [0, 1])
        else:
            for i in range(max(len(q0g), len(q1g))):
                try:
                    qc.append(_single_gate_to_qiskit[q0g[i]], [0])
                except IndexError:
                    qc.append(_single_gate_to_qiskit['I'], [0])
                try:
                    qc.append(_single_gate_to_qiskit[q1g[i]], [1])
                except IndexError:
                    qc.append(_single_gate_to_qiskit['I'], [1])
    return qc


def clifford_to_unitary(gate_seq):
    qc = _clifford_to_qiskit_circ(gate_seq)
    simulator = Aer.get_backend("unitary_simulator")
    unitary: np.ndarray = execute(qc, simulator).result().get_unitary()

    return unitary


if _generate_table:
    print('starting...')
    c2_unitaries = [clifford_to_unitary(index_to_clifford(i)) for i in range(size_c2)]
    np.savez_compressed('c2_unitaries', c2_unitaries)
    print('done')
else:
    ld = np.load('../raw/c2_unitaries.npz')
    c2_unitaries = ld['arr_0']


def is_phase(unitary):
    if np.abs(np.abs(unitary[0, 0]) - 1) < 1e-10:
        if np.max(np.abs(unitary / unitary[0, 0] - np.eye(4))) < 1e-10:
            return True
    else:
        return False


def unitary_to_index(unitary):
    matches = []
    prod_unitaries = unitary.conj().T @ c2_unitaries
    eye4 = np.eye(4)
    for i in range(size_c2):
        if np.abs(np.abs(prod_unitaries[i, 0, 0]) - 1) < 1e-10:
            if np.max(np.abs(prod_unitaries[i] / prod_unitaries[i, 0, 0] - eye4)) < 1e-10:
                matches.append(i)
    assert len(matches) == 1, f"algorithm failed, found {len(matches)} matches > 1"

    return matches[0]


def generate_clifford_truncations(seq_len: int,
                                  truncations_positions: Optional[List] = None,
                                  seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)

    if truncations_positions is None:
        truncations_positions = range(seq_len)
    else:
        set_truncations = set(truncations_positions)
        set_truncations.add(seq_len)
        truncations_positions.append(sorted(list(set_truncations)))

    # generate total sequence:
    main_seq = [index_to_clifford(index) for index in np.random.randint(size_c2, size=seq_len)]
    main_seq_unitaries = [clifford_to_unitary(seq) for seq in main_seq]
    truncations_plus_inverse = []

    # generate truncations:
    for pos in truncations_positions:
        trunc = main_seq[:pos + 1]
        trunc_unitary = main_seq_unitaries[:pos + 1]
        trunc_unitary_prod = np.eye(4)
        for unitary in trunc_unitary:
            trunc_unitary_prod = unitary @ trunc_unitary_prod
        inverse_unitary = trunc_unitary_prod.conj().T
        inverse_clifford = index_to_clifford(unitary_to_index(inverse_unitary))
        trunc.append(inverse_clifford)
        truncations_plus_inverse.append(trunc)
    return truncations_plus_inverse


def _clifford_seq_to_qiskit_circ(clifford_seq):
    qc = QuantumCircuit(2)
    for clifford in clifford_seq:
        qc += _clifford_to_qiskit_circ(clifford)
        qc.barrier()
    return qc


if __name__ == '__main__':
    if _test_2_design:
        sum_2d = 0
        for i in range(size_c2):
            if i % 10 == 0:
                print(i)
            for j in range(size_c2):
                sum_2d += np.abs(np.trace(c2_unitaries[i].conj().T @ c2_unitaries[j])) ** 4
        print("2 design ? ")
        print(sum_2d / size_c2 ** 2)
