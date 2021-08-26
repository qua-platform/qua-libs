from typing import Optional, List, Callable, Tuple, Union, Iterable
import numpy as np
from qualang_tools.bakery.bakery import baking, Baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.QmJob import JobResults

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

ld = np.load('c2_unitaries.npz')
c2_unitaries = ld['arr_0']

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


"""
In what follows, q_tgt should be the main target qubit for which should be played the single qubit gate.
qe_set can be a set of additional quantum elements that might be needed to actually compute the gate 
(e.g fluxline, trigger, ...) 
"""


def I(b: Baking, q_tgt, *qe_set: str):
    pass


def X(b: Baking, q_tgt, *qe_set: str):
    pass


def Y(b: Baking, q_tgt, *qe_set: str):
    pass


def X_2(b: Baking, q_tgt, *qe_set: str):
    pass


def Y_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mX_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mY_2(b: Baking, q_tgt, *qe_set: str):
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
    def __init__(self, *, qmm: QuantumMachinesManager, config: dict, max_length: int, K: int,
                 two_qb_gate_baking_macros: dict[Callable], measure_macro: Callable,
                 measure_args: Optional[Tuple] = None,
                 single_qb_macros: Optional[dict] = None,
                 truncations_positions: Optional[Iterable] = None, seed: Optional[int] = None, quantum_elements: Iterable[str]):
        """
        Class to retrieve easily baked RB sequences and their inverse operations
        Protocol can played by using the method execute()

        :param qmm QuantumMachinesManager instance
        :param config Configuration file
        :param max_length Maximum length of desired RB sequence
        :param K Number of RB sequences
        :param two_qb_gate_baking_macros dictionary containing baking macros for 2 qb gates necessary to do all
        Cliffords (should contain keys "CNOT", "iSWAP" and "SWAP" macros)
        :param measure_macro QUA macro for measurement of the qubits
        :param measure_args arguments to be passed to measure_macro
        :param single_qb_macros baking macros for playing single qubit Cliffords (should contain keys "I", "X", "Y",
        "X/2", "Y/2", "-X/2", "-Y/2")
        :param truncations_positions list containing integers for building RB sequences of varying lengths
        to perform fitting. If no list is provided, RB sequence for each length until max_length is generated
        :param seed Random seed
        :param quantum_elements quantum elements in format (q1, q2, coupler, *other_elements)
        :param seed
        """
        self.qmm = qmm
        self.config = config
        for qe in quantum_elements:
            if qe not in config["elements"]:
                raise KeyError(f"Quantum element {qe} is not in the config")
        if seed is not None:
            np.random.seed(seed)
        if truncations_positions is None:
            self.truncations_positions = range(max_length)
        else:
            set_truncations = set(truncations_positions)
            set_truncations.add(max_length)
            list_truncations = sorted(list(set_truncations))
            self.truncations_positions = list_truncations
        self.measure_macro = measure_macro
        self.measure_args = measure_args
        self.sequences = [
            TwoQbRBSequence(self.qmm, self.config, max_length, two_qb_gate_baking_macros, single_qb_macros,
                            self.truncations_positions, seed, *quantum_elements) for _ in range(K)]

    def qua_prog(self, b_seq: Baking):
        with program() as prog:
            b_seq.run()
            self.measure_macro(self.measure_args)

        return prog

    def execute(self):
        overall_results = {}
        for seq in self.sequences:
            b_seq = seq.generate_baked_sequence()
            prog = self.qua_prog(b_seq=b_seq)
            qm = self.qmm.open_qm(self.config, close_other_machines=True)
            pid = qm.compile(prog)

            for trunc_index in range(len(self.truncations_positions)):
                truncated_wf = seq.generate_baked_truncated_sequence(b_seq, trunc_index)
                pjob = qm.queue.add_compiled(pid, overrides=truncated_wf)
                job = pjob.wait_for_execution()
                results = job.result_handles
                results.wait_for_all_values()
                self.post_process(results, overall_results)
            b_seq.delete_baked_Op()

        return overall_results

    def post_process(self, results: JobResults, overall_results: dict):
        pass


class TwoQbRBSequence:
    def __init__(self, qmm: QuantumMachinesManager, config: dict, d_max: int, two_qubit_gate_macros: dict,
                 single_qb_macros: Optional[dict] = None,
                 truncations_positions: Optional[List] = None, seed: Optional[int] = None, *quantum_elements: str
                 ):
        self.qmm = qmm
        self.d_max = d_max
        self.config = config
        self.truncations_positions = truncations_positions
        self.seed = seed
        assert len(quantum_elements) >= 2, "Two qubit RB requires at least two quantum elements"
        self.quantum_elements = quantum_elements
        self.two_qb_gate_macros = two_qubit_gate_macros
        self.single_qb_macros = single_qb_macros
        self.full_sequence = self.generate_RB_sequence()
        # self.baked_sequence = self.generate_baked_sequence()  # Store the RB sequence
        # self.baked_wf_truncations = [self.generate_baked_truncated_sequence(trunc)
        #                              for trunc in self.truncations_positions] +\
        #                             [self.baked_sequence.get_waveforms_dict()]

    def generate_RB_sequence(self):

        # generate total sequence:
        main_seq = [self.index_to_clifford(index)
                    # List of dictionaries (one key per qe) containing sequences for each Clifford)
                    for index in np.random.randint(low=0, high=size_c2, size=self.d_max)]
        main_seq_unitaries = [self.clifford_to_unitary(seq) for seq in main_seq]  # Unitaries to find inverse op
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

    def _writing_baked_wf(self, b: Baking, trunc):
        q0, q1 = self.quantum_elements[0], self.quantum_elements[1]
        for clifford in self.full_sequence[trunc]:
            assert len(clifford[q0]) == len(clifford[q1])
            for op0, op1 in zip(clifford[q0], clifford[q1]):
                if len(op0) == len(op1):
                    for opa, opb in zip(op0, op1):
                        if opa == "CNOT" or opa == "SWAP" or opa == "iSWAP":
                            assert opa == opb
                            b.align(*self.quantum_elements)
                            self.two_qb_gate_macros[opa](b, *self.quantum_elements)
                        else:
                            self.play_single_qb_op(opa, q0, b)
                            self.play_single_qb_op(opb, q1, b)

                else:
                    for opa in op0:
                        self.play_single_qb_op(opa, q0, b)
                    for opb in op1:
                        self.play_single_qb_op(opb, q1, b)
            b.align(*self.quantum_elements)

    def generate_baked_sequence(self):
        """
        Generates the longest sequence desired with its associated inverse operation.
        The resulting baking object is the reference for overriding baked waveforms that are used for
        truncations, i.e for shorter sequences. Config is updated with this method
        """
        with baking(self.config, padding_method="right", override=True) as b:
            self._writing_baked_wf(b, -1)
        return b

    def generate_baked_truncated_sequence(self, b_ref: Baking, trunc: int):
        """Generate truncated sequences compatible with waveform overriding for add_compiled feature. Config
        is not updated by this method"""

        with baking(self.config, padding_method="right", override=False,
                    baking_index=b_ref.get_baking_index()) as b_new:
            self._writing_baked_wf(b_new, trunc)

        return b_new.get_waveforms_dict()

    def index_to_clifford(self, index):
        """
        Returns a dictionary with list of operations to be conducted to run the Clifford indicated by the index
        for each quantum element
        """
        clifford = {}
        q0 = self.quantum_elements[0]
        q1 = self.quantum_elements[1]
        for qe in self.quantum_elements:
            clifford[qe] = []
        if index < 576:
            # single qubit class
            q0c1, q1c1 = np.unravel_index(index, (24, 24))
            clifford[q0].append(_c1_ops[q0c1])
            clifford[q1].append(_c1_ops[q1c1])

        elif 576 <= index < 576 + 5184:
            # CNOT class
            index -= 576
            q0c1, q1c1, q0s1, q1s1y2 = np.unravel_index(index, (24, 24, 3, 3))
            clifford[q0].append(_c1_ops[q0c1])
            clifford[q1].append(_c1_ops[q1c1])

            clifford[q0].append(("CNOT",))
            clifford[q1].append(("CNOT",))

            clifford[q0].append(_s1_ops[q0s1])
            clifford[q1].append(_s1y2_ops[q1s1y2])

        elif 576 + 5184 <= index < 576 + 2 * 5184:
            # iSWAP class
            index -= 576 + 5184
            q0c1, q1c1, q0s1y2, q1s1x2 = np.unravel_index(index, (24, 24, 3, 3))
            clifford[q0].append(_c1_ops[q0c1])
            clifford[q1].append(_c1_ops[q1c1])

            clifford[q0].append(("iSWAP",))
            clifford[q1].append(("iSWAP",))

            clifford[q0].append(_s1y2_ops[q0s1y2])
            clifford[q1].append(_s1x2_ops[q1s1x2])

        else:
            # swap class
            index -= 576 + 2 * 5184
            q0c1, q1c1 = np.unravel_index(index, (24, 24))

            clifford[q0].append(_c1_ops[q0c1])
            clifford[q1].append(_c1_ops[q1c1])

            clifford[q0].append(("SWAP",))
            clifford[q1].append(("SWAP",))

        return clifford

    def clifford_to_unitary(self, gate_seq):
        unitary: np.ndarray = np.eye(4)
        q0 = self.quantum_elements[0]
        q1 = self.quantum_elements[1]
        g0 = gate_seq[q0]
        g1 = gate_seq[q1]
        assert len(g0) == len(g1), f"Clifford not correctly built, length on q0 ({len(g0)}) != length on q1 ({len(g1)})"
        for c0, c1 in zip(g0, g1):
            if len(c0) == len(c1):
                for opa, opb in zip(c0, c1):
                    if opa == "CNOT" or opa == "SWAP" or opa == "iSWAP":
                        assert opa == opb, f"Two qubit gate does not involve the two elements: " \
                                           f"op for qubit 1 {opa} does not match op for qubit 2 {opb}"
                        unitary = gate_unitaries[opa] @ unitary
                    else:
                        unitary = np.kron(gate_unitaries[opa], gate_unitaries[opb]) @ unitary

            else:
                for opa in c0:
                    unitary = np.kron(gate_unitaries[opa], np.eye(2)) @ unitary
                for opb in c1:
                    unitary = np.kron(np.eye(2), gate_unitaries[opb]) @ unitary
        return unitary

    def play_single_qb_op(self, op: str, q: str, b: Baking):
        if self.single_qb_macros is not None:
            self.single_qb_macros[op](b, q, self.quantum_elements)
        else:
            b.play(op, q)


gate_unitaries = {
    'I': np.eye(2),
    'X': np.array([[0., 1.],
                   [1., 0.]]),
    'Y': np.array([[0., -1j],
                   [1j, 0.]]),
    'X/2': 1 / np.sqrt(2) * np.array([[1., -1j],
                                      [-1j, 1.]]),
    'Y/2': 1 / np.sqrt(2) * np.array([[1., -1.],
                                      [1., 1.]]),
    '-X/2': 1 / np.sqrt(2) * np.array([[1., 1j],
                                      [1j, 1.]]),
    '-Y/2': 1 / np.sqrt(2) * np.array([[1., 1.],
                                      [-1., 1.]]),
    'CNOT': np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.]]),
    'SWAP': np.array([[1., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., 1.]]),
    'iSWAP': np.array([[1., 0., 0., 0.],
                      [0., 0., 1j, 0.],
                      [0., 1j, 0., 0.],
                      [0., 0., 0., 1.]])
}


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
