from typing import List

import numpy as np
from matplotlib import pyplot as plt
from qm import Program, QuantumMachinesManager
from qm.qua import *
from tqdm import tqdm

from .util import run_in_thread
from .TwoQubitRB import TwoQubitRb
from .verification import SequenceTracker


class TwoQubitRbDebugger:
    def __init__(self, rb: TwoQubitRb):
        """
        Class which mimics the preparation, measurement and input streaming techniques
        used in TwoQubitRB to generate a program on a limited subset of gates and
        compares the results to expectation for debugging.
        """
        self.rb = rb

    def run_phased_xz_commands(self, qmm: QuantumMachinesManager, num_averages: int):
        """
        Run a program testing selected commands containing only combinations of PhasedXZ
        gates which lead to a variery of transformations on the |00> staet.
        This is useful for testing the 1Q component of your gate implementation.
        """
        sequences_dict = {
            r"I \otimes I": [720],  # Identity on both qubits
            r"I \otimes Z": [732],  # Z on qubit 1, Identity on qubit 2
            r"Z \otimes I": [723],  # Z on qubit 2, Identity on qubit 1
            r"I \otimes X": [724],  # X on qubit 1, Identity on qubit 2
            r"X \otimes I": [721],  # X on qubit 2, Identity on qubit 1
            r"X \otimes X": [725],  # X on both qubits
            r"\frac{X}{2} \otimes I": [1],  # X/2 on qubit 2, Identity on qubit 1
            r"I \otimes \frac{X}{2}": [6],  # X/2 on qubit 1, Identity on qubit 2
            r"\frac{X}{2} \otimes \frac{X}{2}": [7],  # X/2 on both qubits
            r"\text{CZ}": [74],  # Controlled-Z (CZ) gate
            r"(\frac{X}{2} \otimes \frac{X}{2}), \text{CZ}, (\frac{X}{2} \otimes \frac{X}{2})": [7, 74, 7],
            r"\text{CNOT}": [12, 347],  # X/2 on qubit 2, followed by CNOT
            r"(\frac{X}{2} \otimes I), \text{CNOT}": [1, 4, 63],  # X/2 on qubit 2, followed by CNOT
            r"(X \otimes I), \text{CNOT}": [724, 4, 63],  # X/2 on qubit 2, followed by CNOT
            r"(I \otimes X), \text{SWAP}": [724, 39, 489],  # X on qubit 1, followed by SWAP
        }

        sequences = sequences_dict.values()

        self.sequence_tracker = SequenceTracker(self.rb._command_registry)

        prog = self._phased_xz_commands_program(len(sequences), num_averages)

        qm = qmm.open_qm(self.rb._config)
        job = qm.execute(prog)

        for sequence in tqdm(sequences, desc='Running test-sequences', unit='sequence'):
            self.sequence_tracker.make_sequence(sequence)
            self._insert_all_input_stream(job, sequence)

        job.result_handles.wait_for_all_values()
        state = job.result_handles.get("state").fetch_all()

        self.sequence_tracker.print_sequences()
        self._analyze_phased_xz_commands_program(state, list(sequences_dict.keys()))

    @run_in_thread
    def _insert_all_input_stream(self, job, sequence):
        job.insert_input_stream("__gates_len_is__", len(sequence))
        for qe in self.rb._rb_baker.all_elements:
            job.insert_input_stream(f"{qe}_is", self.rb._decode_sequence_for_element(qe, sequence))

    def _phased_xz_commands_program(self, num_sequences: int, num_averages: int) -> Program:
        with program() as prog:
            n_avg = declare(int)
            state = declare(int)
            length = declare(int)
            i = declare(int)
            state_os = declare_stream()
            gates_len_is = declare_input_stream(int, name="__gates_len_is__", size=1)
            gates_is = {
                qe: declare_input_stream(int, name=f"{qe}_is", size=self.rb._buffer_length)
                for qe in self.rb._rb_baker.all_elements
            }

            with for_(i, 0, i < num_sequences, i + 1):
                advance_input_stream(gates_len_is)
                for gate_is in gates_is.values():
                    advance_input_stream(gate_is)
                assign(length, gates_len_is[0])
                with for_(n_avg, 0, n_avg < num_averages, n_avg + 1):
                    self.rb._prep_func()
                    self.rb._rb_baker.run(gates_is, length)
                    out1, out2 = self.rb._measure_func()
                    assign(state, (Cast.to_int(out2) << 1) + Cast.to_int(out1))
                    save(state, state_os)

            with stream_processing():
                state_os.buffer(num_sequences, num_averages).save("state")

        return prog

    def _analyze_phased_xz_commands_program(self, state: np.ndarray, sequence_labels: List[str]):
        fig, axs = plt.subplots(5, 3, figsize=(12, 10))
        axs = axs.ravel()

        basis_states = [fr"$|00\rangle$", fr"$|01\rangle$", fr"$|10\rangle$", fr"$|11\rangle$"]

        for i, sequence in enumerate(self.sequence_tracker._sequences_as_gates):
            expected_state = self.sequence_tracker.calculate_resultant_state(sequence)
            expected_distribution_into_two_qubit_bases = np.diag(expected_state)

            counts = np.bincount(state[i], minlength=4)
            counts = counts / counts.sum()

            axs[i].set_title(f"${sequence_labels[i]}$")
            axs[i].bar(basis_states, expected_distribution_into_two_qubit_bases, label='Expected', alpha=0.5, color='b')
            axs[i].bar(basis_states, counts, label='Measured', alpha=0.5, color='r')

        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', ncol=2)
        fig.suptitle(r"State Distribution of PhasedXZ Commands applied to $|00\rangle$")
        fig.supylabel("Measurement Probability")
        fig.supxlabel(r"Basis State as $|q_2, q_1\rangle$")

        plt.tight_layout()
        plt.plot()
        plt.show()
