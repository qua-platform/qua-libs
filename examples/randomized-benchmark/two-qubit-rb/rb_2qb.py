from typing import Optional, List

from qiskit import QuantumCircuit, Aer, execute
import qiskit.circuit.library as glib
import numpy as np
from configuration import config
from qualang_tools.bakery.bakery import baking, Baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

# this builds up on https://arxiv.org/pdf/1402.4848
# build up sequence

_generate_table = False
_test_2_design = False  # careful, this takes about 20 minutes
size_c2 = 11520

"""If no single_qb_macros are provided, Single qubit Clifford generators are assumed to be already setup in the config with 
operations names matching ops described below 
(i.e 'I', 'X', 'Y', 'Y/2', 'X/2', '-X/2' and '-Y/2' are present in operations 
list of both elements describing qubit 0 and qubit 1"""

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

"""
For two qubit gates necessary to generate the 4 classes (see Supplementary info of this paper:
https://arxiv.org/pdf/1210.7011.pdf), we require the user to complete the following macros below according
to their own native set of qubit gates, that is perform the appropriate decomposition and convert the pulse sequence
in a sequence of baking play statements (amounts to similar structure as QUA, just add the prefix b. before every statement)
Example :
in QUA you would have for a CZ operation:
    play("CZ", "coupler")
in the macros below you write instead:
    b.play("CZ", "coupler")
"""


# Baking Macros needed for two qubit gates

def CNOT(b: Baking, *qe_set: str):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #
    pass


def iSWAP(b: Baking, *qe_set: str):
    pass


def SWAP(b: Baking, *qe_set: str):
    pass


def add_single_qubit_clifford(clifford, index, q):
    for op in _c1_ops[index]:
        clifford[q].append(op)


def add_s_op(s1_ops, clifford, index, q):
    for op in s1_ops[index]:
        clifford[q].append(op)


def I(b: Baking, *qe_set: str):
    pass


def X(b: Baking, *qe_set: str):
    pass


def Y(b: Baking, *qe_set: str):
    pass


def X_2(b: Baking, *qe_set: str):
    pass


def Y_2(b: Baking, *qe_set: str):
    pass


def mX_2(b: Baking, *qe_set: str):
    pass


def mY_2(b: Baking, *qe_set: str):
    pass


two_qb_gate_macros = {
    "CNOT": CNOT,
    "iSWAP": iSWAP,
    "SWAP": SWAP
}

single_qb_gate_macros = {
    "I": I,
    "X": X,
    "Y": Y,
    "X/2": X_2,
    "-X/2": mX_2,
    "Y/2": Y_2,
    "-Y/2": mY_2
}


class RBTwoQubits:
    def __init__(self, qmm: QuantumMachinesManager, config: dict, max_length: int, K: int,
                 two_qb_gate_macros: dict, measure_macro: function, single_qb_macros: Optional[dict] = None,
                 truncations_positions: Optional[List] = None, seed: int = None, *quantum_elements: str):
        """
        Class to retrieve easily baked RB sequences and their inverse operations
        :param qmm QuantumMachinesManager instance
        :param config Configuration file
        :param max_length Maximum length of desired RB sequence
        :param K Number of RB sequences
        :param quantum_elements quantum elements in format (q1, q2, coupler, *other_elements)
        :param truncations_positions
        :param seed
        """
        self.qmm = qmm
        for qe in quantum_elements:
            if qe not in config["elements"]:
                raise KeyError(f"Quantum element {qe} is not in the config")

        self.sequences = [TwoQbRBSequence(self.qmm, config, max_length, two_qb_gate_macros, single_qb_macros,
                                          truncations_positions, seed, *quantum_elements) for _ in range(K)]
        self.inverse_ops = [seq.revert_ops for seq in self.sequences]
        self.duration_trackers = [seq.duration_tracker for seq in self.sequences]
        self.baked_sequences = [seq.full_sequence for seq in self.sequences]

    def QUA_prog(self, seq: Baking):
        with program() as prog:
            seq.run()

        return prog

    def execute(self):
        for seq in self.baked_sequences:
            prog = self.QUA_prog(seq)
            qm = self.qmm.open_qm(self.config, close_other_machines=True)
            pid = qm.queue.compile(prog)

        pjob = qm.queue.add_compiled(prog, overrides={b_new.get_waveforms_dict()})
        job = qm.queue.wait_for_execution(pjob)
        job.results_handles.wait_for_all_values()

class TwoQbRBSequence:
    def __init__(self, qmm: QuantumMachinesManager, config: dict, d_max: int, two_qb_gate_macros: dict, single_qb_macros: Optional[dict] = None,
                 truncations_positions: Optional[List] = None, seed: Optional[int] = None, *quantum_elements: str
                 ):
        self.qmm = qmm
        self.d_max = d_max
        self.config = config
        self.truncations_positions = truncations_positions
        self.seed = seed
        assert len(quantum_elements) >= 2, "Two qubit RB requires at least two quantum elements"
        self.quantum_elements = quantum_elements
        self.two_qb_gate_macros = two_qb_gate_macros
        self.single_qb_macros = single_qb_macros
        self.full_sequence = self.generate_RB_sequence()
        self.duration_tracker = [0] * d_max  # Keeps track of each Clifford's duration
        self.operations_list = [None] * d_max
        self.inverse_op_string = [""] * d_max
        self.baked_sequence = self.generate_baked_sequence()  # Store the RB sequence
        self.baked_wf_truncations = [self.generate_baked_truncated_sequence(trunc)
                                     for trunc in self.truncations_positions]

    def generate_RB_sequence(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.truncations_positions is None:
            self.truncations_positions = range(self.d_max)
        else:
            set_truncations = set(self.truncations_positions)
            set_truncations.add(self.d_max)
            self.truncations_positions.append(sorted(list(set_truncations)))

        # generate total sequence:
        main_seq = [self.index_to_clifford(index)
                    # List of dictionaries (one key per qe) containing sequences for each Clifford)
                    for index in np.random.randint(low=0, high=size_c2, size=self.d_max)]
        main_seq_unitaries = [clifford_to_unitary(seq) for seq in main_seq]  # Unitaries to find inverse op
        truncations_plus_inverse = []

        # generate truncations:
        for pos in self.truncations_positions:
            trunc = main_seq[:pos + 1]
            trunc_unitary = main_seq_unitaries[:pos + 1]
            trunc_unitary_prod = np.eye(4)
            for unitary in trunc_unitary:
                trunc_unitary_prod = unitary @ trunc_unitary_prod
            inverse_unitary = trunc_unitary_prod.conj().T
            inverse_clifford = self.index_to_clifford(unitary_to_index(inverse_unitary))
            trunc.append(inverse_clifford)
            truncations_plus_inverse.append(trunc)
        return truncations_plus_inverse

    def writing_baked_wf(self, b: Baking, trunc):
        for clifford in self.full_sequence[trunc]:
            for qe in clifford:
                for op in clifford[qe]:
                    if op == "CNOT" or op == "SWAP" or op == "iSWAP":
                        self.two_qb_gate_macros[op](b, self.quantum_elements)
                    else:
                        if self.single_qb_macros is not None:
                            self.single_qb_macros[op](b, self.quantum_elements)
                        else:
                            b.play(op, qe)
            b.align(*self.quantum_elements)

    def generate_baked_sequence(self):
        """
        Generates the longest sequence desired with its associated inverse operation.
        The resulting baking object is the reference for overriding baked waveforms that are used for
        truncations, i.e for shorter sequences
        """
        with baking(self.config, padding_method="right", override=True) as b:
            self.writing_baked_wf(b, -1)
        return b

    def generate_baked_truncated_sequence(self, trunc):
        """Generate truncated sequences compatible with waveform overriding for add_compiled feature"""

        with baking(self.config, padding_method="right", override=False,
                    baking_index=self.baked_sequence.get_baking_index()) as b_new:

            self.writing_baked_wf(b_new, trunc)

        return b_new.get_waveforms_dict()

    def index_to_clifford(self, index):
        clifford = {}
        q0 = self.quantum_elements[0]
        q1 = self.quantum_elements[1]
        for qe in self.quantum_elements:
            clifford[qe] = []
        if index < 576:
            # single qubit class
            q0c1, q1c1 = np.unravel_index(index, (24, 24))

            add_single_qubit_clifford(clifford, q0c1, q0)
            add_single_qubit_clifford(clifford, q1c1, q1)

        elif 576 <= index < 576 + 5184:
            # CNOT class
            index -= 576
            q0c1, q1c1, q0s1, q1s1y2 = np.unravel_index(index, (24, 24, 3, 3))
            add_single_qubit_clifford(clifford, q0c1, q0)
            add_single_qubit_clifford(clifford, q1c1, q1)
            for q in self.quantum_elements:
                clifford[q].append("CNOT")
            add_s_op(_s1_ops, clifford, q0s1, q0)
            add_s_op(_s1y2_ops, clifford, q1s1y2, q1)

        elif 576 + 5184 <= index < 576 + 2 * 5184:
            # iSWAP class
            index -= 576 + 5184
            q0c1, q1c1, q0s1y2, q1s1x2 = np.unravel_index(index, (24, 24, 3, 3))
            add_single_qubit_clifford(clifford, q0c1, q0)
            add_single_qubit_clifford(clifford, q1c1, q1)
            for q in self.quantum_elements:
                clifford[q].append("iSWAP")
            add_s_op(_s1y2_ops, clifford, q0s1y2, q0)
            add_s_op(_s1x2_ops, clifford, q1s1x2, q1)

        else:
            # swap class
            index -= 576 + 2 * 5184
            q0c1, q1c1 = np.unravel_index(index, (24, 24))
            for q in self.quantum_elements:
                clifford[q].append("SWAP")
            add_single_qubit_clifford(clifford, q0c1, q0)
            add_single_qubit_clifford(clifford, q1c1, q1)

        return clifford


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
