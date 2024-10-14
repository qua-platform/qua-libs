from typing import List

import numpy as np
from matplotlib import pyplot as plt
from qm import Program, QuantumMachinesManager
from qm.qua import *
from tqdm import tqdm

from CS_installations.quam_libs.experiments.two_qubit_rb.util import run_in_thread
from TwoQubitRB import TwoQubitRb
from verification import SequenceTracker


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
        Run a program testing all commands containing only combinations of PhasedXZ
        gates. This is useful for testing the 1Q component of your gate implementation.
        """
        commands = range(35)
        self.sequence_tracker = SequenceTracker(self.rb._command_registry)

        prog = self._phased_xz_commands_program(commands, num_averages)

        qm = qmm.open_qm(self.rb._config)
        job = qm.execute(prog)

        for command in tqdm(commands, desc='Running test-commands', unit='command'):
            sequence = [command]
            self.sequence_tracker.make_sequence(sequence)
            self._insert_all_input_stream(job, sequence)

        job.result_handles.wait_for_all_values()
        state = job.result_handles.get("state").fetch_all()

        self.sequence_tracker.print_sequences()
        self._analyze_phased_xz_commands_program(state)

    @run_in_thread
    def _insert_all_input_stream(self, job, sequence):
        job.insert_input_stream("__gates_len_is__", len(sequence))
        for qe in self.rb._rb_baker.all_elements:
            job.insert_input_stream(f"{qe}_is", self.rb._decode_sequence_for_element(qe, sequence))

    def _phased_xz_commands_program(self, commands: List[int], num_averages: int) -> Program:
        with program() as prog:
            n_avg = declare(int)
            state = declare(int)
            length = declare(int)
            state_os = declare_stream()
            gates_len_is = declare_input_stream(int, name="__gates_len_is__", size=1)
            gates_is = {
                qe: declare_input_stream(int, name=f"{qe}_is", size=self.rb._buffer_length)
                for qe in self.rb._rb_baker.all_elements
            }

            for _ in range(len(commands)):
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
                state_os.buffer(len(commands), num_averages).save("state")

        return prog

    def _analyze_phased_xz_commands_program(self, state: np.ndarray):
        fig, axs = plt.subplots(7, 5)
        axs = axs.ravel()
        for i, sequence in self.sequence_tracker._sequences_as_gates:
            expected_state = self.sequence_tracker.calculate_resultant_state(sequence)
            expected_distribution_into_two_qubit_bases = np.diag(expected_state)
            axs[i].set_title(f"Sequence {i}")
            axs[i].hist(state, label='Measured')
            axs[i].hist(expected_distribution_into_two_qubit_bases, label='Expected', alpha=0.5)
            axs[i].legend()

        plt.plot()
        plt.tight_layout()