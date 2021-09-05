from typing import Optional, List, Callable, Tuple, Union, Dict
import numpy as np
# from qualang_tools.bakery.bakery import baking, Baking
from bakery.bakery import baking, Baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from copy import deepcopy

# this builds up on https://arxiv.org/pdf/1402.4848
# build up sequence

_generate_table = False
_test_2_design = False  # careful, this takes about 20 minutes
size_c2 = 11520
Clifford = Dict[str, List[Tuple]]
"""If no single_qb_macros are provided, Single qubit Clifford generators are assumed 
to be already setup in the config with operations names matching ops described below 
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


def retrieve_truncations(seq, baked_reference, trunc_index):
    return seq.retrieve_truncations(baked_reference, trunc_index)


class RBTwoQubits:
    def __init__(self, qmm: QuantumMachinesManager,
                 config: Dict,
                 N_Clifford: Union[Iterable, int],
                 K: int,
                 two_qb_gate_baking_macros: Dict[str, Callable],
                 quantum_elements: Iterable[str] = None,
                 single_qb_macros: Optional[Dict[str, Callable]] = None,
                 seed: Optional[int] = None
                 ):
        """
        Class designed to ease the realization of a Two-qubit Randomized Benchmarking experiment.
        The class generates sequences of 2 qubit Cliffords (selection done in four classes as done in
        https://arxiv.org/abs/1210.7011), and creates a series of baked waveforms (one per random sequence) and plays
        them successively with the use of one single QUA program.
        Shorter random sequences are also generated from truncations and are played
        more efficiently from the original baked waveform using the add_compiled feature of the QM API.

        The user is expected to provide macros for two qubit Clifford generators (CNOT, iSWAP and SWAP) based on their
        own set of native gates,
        Additional parameters can be provided such as truncation positions, macros for single qubit gates if those are
        more complex than one single QUA play statement and a random seed.
        Protocol can played by using the method execute()


        :param qmm: QuantumMachinesManager instance
        :param config: Configuration file
        :param N_Clifford:
            Number of Clifford gates per sequence. If iterable is provided, then a series of sequences of various
            lengths are generated. If integer is provided, all sequence truncations are generated up to
            the indicated max number.
        :param K: Number of RB sequences

        :param two_qb_gate_baking_macros:
            dictionary containing baking macros for 2 qb gates necessary to do all
            Cliffords (should contain keys "CNOT", "iSWAP" and "SWAP" macros)

        :param single_qb_macros:
            baking macros for playing single qubit Cliffords (should contain keys "I", "X", "Y",
            "X/2", "Y/2", "-X/2", "-Y/2"). If None is provided, then the param quantum_elements should carry the name of
            the quantum elements representing qubit 0 and 1.

        :param seed: Random seed

        :param quantum_elements:
            quantum elements involved for qubit 0 and 1, should be the same name as in the config. If none is provided,
            then macros for single qubit gates are required
        """

        self.qmm = qmm
        self.config = config
        if quantum_elements is not None:
            for qe in quantum_elements:
                if qe not in config["elements"]:
                    raise KeyError(f"Quantum element {qe} is not in the config")
        else:
            if single_qb_macros is None:
                raise KeyError("No elements or single qubit gate macros have been provided,"
                               " impossible to do single qubit gates")
        if seed is not None:
            np.random.seed(seed)
        if type(N_Clifford) == int:
            self.N_Clifford = range(N_Clifford)
        else:
            set_truncations = set(N_Clifford)
            list_truncations = sorted(list(set_truncations))
            self.N_Clifford = list_truncations
        self.sequences = [
            TwoQbRBSequence(self.qmm, config,
                            self.N_Clifford,
                            two_qb_gate_baking_macros,
                            single_qb_macros,
                            seed,
                            quantum_elements) for _ in range(K)]

        max_length = 0
        tgt_seq = None

        for seq in self.sequences:
            if max_length < seq.sequence_length:
                max_length = seq.sequence_length
                tgt_seq = seq
        self.config.update(tgt_seq.mock_config)
        self.baked_reference = tgt_seq.baked_sequence

    def run(self, prog):
        """
        Run the full two qubit RB experiment

        :param prog: QUA program to be executed, should carry the following form:
            RB_EXP = RBTwoQubits(...)
            b = RB_EXP.baked_reference
            with program as prog():
                n = declare(int)
                with for_(n, 0, n < N_shots, n+1):
                    b.run()
                    measure(...)

                with stream_processing():
                    ...
        """
        qm = self.qmm.open_qm(self.config, close_other_machines=True)
        pid = qm.compile(prog)
        results_list = []
        for seq in self.sequences:
            for trunc_index in range(len(self.N_Clifford)):
                truncated_wf = retrieve_truncations(seq, self.baked_reference, trunc_index)
                pending_job = qm.queue.add_compiled(pid, overrides=truncated_wf)
                job = pending_job.wait_for_execution()
                results = job.result_handles
                results.wait_for_all_values()
                results_list.append(results)

        print("Experiment done, results are available")
        return results_list


class TwoQbRBSequence:
    def __init__(self, qmm: QuantumMachinesManager, config: dict,
                 N_Cliffords: List,
                 two_qubit_gate_macros: Dict[str, Callable],
                 single_qb_macros: Optional[Dict[str, Callable]] = None,
                 seed: Optional[int] = None,
                 quantum_elements: Optional[Iterable[str]] = None
                 ):
        self.qmm = qmm
        self.mock_config = deepcopy(config)
        self.truncations_positions = N_Cliffords
        self.d_max = N_Cliffords[-1]
        self.seed = seed
        self._number_of_gates = 0
        if quantum_elements is not None:
            self.quantum_elements = quantum_elements
        else:
            self.quantum_elements = ("q0", "q1")
        self.two_qb_gate_macros = two_qubit_gate_macros
        self.single_qb_macros = single_qb_macros
        self.full_sequence = self.generate_RB_sequence()
        self.baked_sequence = self._generate_baked_sequence()
        self.sequence_length = self.baked_sequence.get_Op_length()

    def generate_RB_sequence(self) -> List[List[Clifford]]:
        """
        Generate all sequences according to provided truncations. Each Clifford is a dict with quantum elements as keys
        and to each key is associated a list of tuples (each tuple is either a single qubit Clifford,
        or a two-qubit gate marker).

        :returns: List of all RB sequences containing Clifford operations in the form:
            [ List of Clifford for truncation #1, List of Clifford for truncation #2,...]
        """

        # generate total sequence:
        main_seq = [self.index_to_clifford(index)
                    # List of dictionaries (one key per qe) containing sequences for each Clifford)
                    for index in np.random.randint(low=0, high=size_c2, size=self.d_max)]
        main_seq_unitaries = [self.clifford_to_unitary(seq) for seq in main_seq]  # Unitaries to find inverse op
        truncations_plus_inverse = []

        # generate truncations:
        for pos in self.truncations_positions:
            trunc = main_seq[:pos+1]
            trunc_unitary = main_seq_unitaries[:pos+1]
            trunc_unitary_prod = np.eye(4)
            for unitary in trunc_unitary:
                trunc_unitary_prod = unitary @ trunc_unitary_prod
            inverse_unitary = trunc_unitary_prod.conj().T
            inverse_clifford = self.index_to_clifford(unitary_to_index(inverse_unitary))
            trunc.append(inverse_clifford)
            truncations_plus_inverse.append(trunc)

        return truncations_plus_inverse

    def _writing_baked_wf(self, b: Baking, trunc) -> None:
        q0, q1 = self.quantum_elements[0], self.quantum_elements[1]
        for clifford in self.full_sequence[trunc]:
            assert len(clifford[q0]) == len(clifford[q1])
            for op0, op1 in zip(clifford[q0], clifford[q1]):
                if len(op0) == len(op1):
                    for opa, opb in zip(op0, op1):
                        if opa == "CNOT" or opa == "SWAP" or opa == "iSWAP":
                            assert opa == opb
                            self.two_qb_gate_macros[opa](b)
                        else:
                            self.play_single_qb_op(opa, q0, b)
                            self.play_single_qb_op(opb, q1, b)

                else:
                    for opa in op0:
                        self.play_single_qb_op(opa, q0, b)
                    for opb in op1:
                        self.play_single_qb_op(opb, q1, b)

    def _generate_baked_sequence(self) -> Baking:
        """
        Generates the longest sequence desired with its associated inverse operation.
        The resulting baking object is the reference for overriding baked waveforms that are used for
        truncations, i.e for shorter sequences. Config is updated with this method

        :returns: Baking object containing RB sequence of longest size
        """
        with baking(self.mock_config, padding_method="right", override=True) as b:
            self._writing_baked_wf(b, -1)
        return b

    def retrieve_truncations(self, b_ref: Baking, trunc: int) -> Dict:
        """Generate truncated sequences compatible with waveform overriding for add_compiled feature. Config
        is not updated by this method"""

        with baking(self.mock_config, padding_method="right", override=False,
                    baking_index=b_ref.get_baking_index()) as b_new:
            self._writing_baked_wf(b_new, trunc)

        return b_new.get_waveforms_dict()

    def index_to_clifford(self, index) -> Clifford:
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
    for x in range(size_c2):
        if np.abs(np.abs(prod_unitaries[x, 0, 0]) - 1) < 1e-10:
            if np.max(np.abs(prod_unitaries[x] / prod_unitaries[x, 0, 0] - eye4)) < 1e-10:
                matches.append(x)
    assert len(matches) == 1, f"algorithm failed, found {len(matches)} matches > 1"

    return matches[0]

