from typing import Optional, List, Callable, Tuple, Union, Dict
import numpy as np
from qualang_tools.bakery.bakery import baking, Baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.QmJob import JobResults
from copy import deepcopy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

size_c2 = 11520
Clifford = Dict[str, List[Tuple]]

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


class RBTwoQubits:
    def __init__(self, qmm: QuantumMachinesManager,
                 config: Dict,
                 N_Clifford: Union[Iterable, int],
                 N_sequences: int,
                 two_qb_gate_baking_macros: Dict[str, Callable],
                 single_qb_macros: Dict[str, Callable],
                 qubit_register: Dict,
                 seed: Optional[int] = None
                 ):
        """
        Class designed for running a Two-qubit Randomized Benchmarking experiment.
        The class generates sequences of 2 qubit Clifford operations (selection done within four classes as done in
        https://arxiv.org/abs/1210.7011), creates a series of baked waveforms (one per random sequence), and returns one
        as a reference to be run within a QUA program prepared by the user.
        Shorter random sequences are also generated from truncations and are played
        more efficiently from the original baked waveform using the add_compiled feature of the QM API.

        The user is expected to provide macros for two qubit Clifford generators (CNOT, iSWAP and SWAP) based on their
        own set of native gates, as well as macros for single qubit Clifford generators (I, X, Y , Y/2, -Y/2, X/2, -X/2
        Additional parameters can be provided such as truncation positions, and a random seed.
        Protocol can played by using the method run()


        :param qmm: QuantumMachinesManager instance
        :param config: Configuration file
        :param N_Clifford:
            Number of Clifford gates per sequence. If iterable is provided, truncations of indicated lengths are run.
            If integer is provided, all truncations are generated up to
            the indicated max number.
        :param N_sequences: Number of RB sequences

        :param two_qb_gate_baking_macros:
            dictionary containing baking macros for 2 qb gates necessary to do all
            Cliffords (should contain 3 keys "CNOT", "iSWAP" and "SWAP" macros)

        :param single_qb_macros:
            baking macros for playing single qubit Cliffords (should contain 7 keys "I", "X", "Y",
            "X/2", "Y/2", "-X/2", "-Y/2").
        :param qubit_register: Aliases dictionary used for Resolver initialization (cf example)
        :param seed: Random seed
        """
        self.qubits = tuple(qubit_register.keys())
        self.qmm = qmm
        self._experiment_completed = False
        self._statistics_retrieved = False
        self._P_00 = 0
        self.alpha = 0
        self.A = 0
        self.B = 0
        self.N_sequences = N_sequences
        if seed is not None:
            np.random.seed(seed)
        if type(N_Clifford) == int:
            self.N_Clifford = range(N_Clifford)
        else:
            set_truncations = set(N_Clifford)
            list_truncations = sorted(list(set_truncations))
            self.N_Clifford = list_truncations

        # Check if macros dictionary contain all required gates
        assert len(two_qb_gate_baking_macros.keys()) == 3, f"{len(two_qb_gate_baking_macros.keys())} provided " \
                                                           f"instead of 3 (CNOT, SWAP and iSWAP)"
        assert "CNOT" in two_qb_gate_baking_macros, "CNOT key not found"
        assert "SWAP" in two_qb_gate_baking_macros, "SWAP key not found"
        assert "iSWAP" in two_qb_gate_baking_macros, "iSWAP key not found"

        assert len(single_qb_macros.keys()) == 7
        assert "I" in single_qb_macros, "I key not found"
        assert "X" in single_qb_macros, "X key not found"
        assert "Y" in single_qb_macros, "Y key not found"
        assert "X/2" in single_qb_macros, "X/2 key not found"
        assert "Y/2" in single_qb_macros, "Y/2 key not found"
        assert "-X/2" in single_qb_macros, "-X/2 key not found"
        assert "-Y/2" in single_qb_macros, "-Y/2 key not found"

        self.sequences = [
            TwoQbRBSequence(self.qmm, config,
                            self.N_Clifford,
                            two_qb_gate_baking_macros,
                            single_qb_macros,
                            self.qubits,
                            seed) for _ in range(N_sequences)]

        max_length = 0
        tgt_seq = None

        for seq in self.sequences:
            if max_length < seq.sequence_length:
                max_length = seq.sequence_length
                tgt_seq = seq
        self.config = deepcopy(tgt_seq.mock_config)
        self.baked_reference = tgt_seq.baked_sequence
        self.results = [[JobResults] * len(self.N_Clifford)] * N_sequences
        self.job_list = []

    def retrieve_truncations(self, seq, baked_reference, trunc_index):
        return seq.retrieve_truncations(self.config, baked_reference, trunc_index)

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
        print("Total number of sequences:", len(self.sequences))
        print("Total number of truncations:", len(self.N_Clifford))
        print("Total number of circuits to be run:", len(self.sequences)*len(self.N_Clifford))
        for i, seq in enumerate(self.sequences):
            print("Running RB for sequence", i)
            for trunc_index in range(len(self.N_Clifford)):
                print(f"Running sequence of {self.N_Clifford[trunc_index]} Cliffords")
                print("Number of Cliffords", len(seq.full_sequence[trunc_index]), seq.full_sequence[trunc_index])

                truncated_wf = self.retrieve_truncations(seq, self.baked_reference, trunc_index)
                pending_job = qm.queue.add_compiled(pid, overrides=truncated_wf)
                job = pending_job.wait_for_execution()
                self.job_list.append(job)
                results = job.result_handles
                results.wait_for_all_values()
                self.results[i][trunc_index] = results

        self._experiment_completed = True
        print("Experiment done, results are available")

    def retrieve_results(self, stream_name_0: str, stream_name_1: str, N_shots: int):
        """
        Retrieves the average error per Clifford issued from the fitting of obtained results
        """
        if self._experiment_completed:
            P_00 = [0.] * len(self.N_Clifford)
            for trunc in range(len(self.N_Clifford)):

                for seq in range(self.N_sequences):
                    results_q0 = self.results[seq][trunc].get(name=stream_name_0).fetch_all()['value']
                    results_q1 = self.results[seq][trunc].get(name=stream_name_1).fetch_all()['value']
                    assert len(results_q0) == len(results_q1), "The two streams provided do not have the same length"
                    assert len(results_q0) == N_shots, "Number of shots provided does not match the length of the streamed data"
                    for i in range(N_shots):
                        if results_q0[i] == 0 and results_q1[i] == 0:
                            P_00[trunc] += 1 / (N_shots*self.N_sequences)
            self._P_00 = P_00
            xdata = self.N_Clifford  # depths
            ydata = P_00
            assert len(xdata) == len(ydata)

            def model(x, a, beta, b):
                return a + b * beta ** x

            popt, pcov = curve_fit(model, xdata, ydata)
            A, alpha, B = popt

            print("Average Error per Clifford: ", 3 * (1.0 - alpha) / 4)
            self.alpha = alpha
            self.A = A
            self.B = B
            self._statistics_retrieved = True
            return P_00, 3 * (1.0 - alpha) / 4
        else:
            raise ValueError("Experiment was not executed (use run() method)")

    def plot(self):
        """
        Plots the experimental results and the fitting function performed through the method retrieve_results
        """
        if self._experiment_completed and self._statistics_retrieved:
            xdata = np.array(self.N_Clifford)
            ydata = np.array(self._P_00)
            plt.figure()
            plt.plot(xdata, ydata, 'x')
            plt.plot(xdata, self.A + self.B * self.alpha**xdata)
            plt.xlabel("Number of Clifford operations")
            plt.ylabel("Average |00> state fidelity")

        else:
            raise NotImplementedError("Plotting not possible: Experiment not completed "
                                      "or method to retrieve results not called")
        
    def retrieve_average_error_gate(self):
        N_gates_per_Clifford = 3
        return 3*(1.0 - self.alpha)/(4 * N_gates_per_Clifford)


class TwoQbRBSequence:
    def __init__(self, qmm: QuantumMachinesManager, config: dict,
                 N_Cliffords: List,
                 two_qubit_gate_macros: Dict[str, Callable],
                 single_qb_macros: Dict[str, Callable],
                 qubit_register: Tuple,
                 seed: Optional[int] = None,
                 ):
        self.qmm = qmm
        self.mock_config = deepcopy(config)
        self.truncations_positions = N_Cliffords
        self.d_max = N_Cliffords[-1]
        self.seed = seed
        self._number_of_gates = 0
        assert len(qubit_register) == 2, "Register shall contain 2 qubits only"
        assert isinstance(qubit_register[0], str)
        assert isinstance(qubit_register[1], str)
        self.qubits = qubit_register
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
            assert np.round(trunc_unitary_prod @ inverse_unitary, 6).all() == np.eye(4).all()
            inverse_clifford = self.index_to_clifford(unitary_to_index(inverse_unitary))
            trunc.append(inverse_clifford)
            truncations_plus_inverse.append(trunc)

        return truncations_plus_inverse

    def _writing_baked_wf(self, b: Baking, trunc) -> None:
        q0, q1 = self.qubits[0], self.qubits[1]
        for clifford in self.full_sequence[trunc]:
            assert len(clifford[q0]) == len(clifford[q1])
            for op0, op1 in zip(clifford[q0], clifford[q1]):
                if len(op0) == len(op1):
                    for opa, opb in zip(op0, op1):
                        if opa == "CNOT" or opa == "SWAP" or opa == "iSWAP":
                            assert opa == opb
                            self.two_qb_gate_macros[opa](b, q0, q1)
                        else:
                            self.single_qb_macros[opa](b, q0)
                            self.single_qb_macros[opb](b, q1)

                else:
                    for opa in op0:
                        self.single_qb_macros[opa](b, q0)
                    for opb in op1:
                        self.single_qb_macros[opb](b, q1)

    def _generate_baked_sequence(self) -> Baking:
        """
        Generates the longest sequence desired with its associated inverse operation.
        The resulting baking object is the reference for overriding baked waveforms that are used for
        truncations, i.e for shorter sequences. Config is updated with this method

        :returns: Baking object containing RB sequence of longest size
        """
        with baking(self.mock_config, padding_method="left", override=True) as b:
            self._writing_baked_wf(b, -1)
            b.align()
        return b

    def retrieve_truncations(self, config: Dict, b_ref: Baking, trunc: int) -> Dict:
        """Generate truncated sequences compatible with waveform overriding for add_compiled feature. Config
        is not updated by this method"""

        with baking(config, padding_method="left", override=False,
                    baking_index=b_ref.get_baking_index()) as b_new:
            self._writing_baked_wf(b_new, trunc)
            b_new.align()

        return b_new.get_waveforms_dict()

    def index_to_clifford(self, index) -> Clifford:
        """
        Returns a dictionary with list of operations to be conducted to run the Clifford indicated by the index
        for each quantum element
        """
        clifford = {}
        q0 = self.qubits[0]
        q1 = self.qubits[1]
        for qe in self.qubits:
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
        q0 = self.qubits[0]
        q1 = self.qubits[1]
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
