from typing import Union

import numpy as np

from .command_registry import *


class SequenceTracker:
    """
    Tracks a randomly-generated sequence by recording both a list of qubit gates
    corresponding to the sequence, and the raw command IDs which are used as
    input to the input stream to map into baked pulses.
    """

    def __init__(self, command_registry: CommandRegistry):
        self.command_registry: CommandRegistry = command_registry
        self._sequences_as_gates: list[Command] = []
        self._sequences_as_command_ids: list[list[int]] = []

    def make_sequence(self, command_ids: list[int]):
        """
        Adds a new sequence to the tracker, recorded as both a list
        of gates and command ids
        """
        self._record_sequence_as_list_of_command_ids(command_ids)
        self._record_sequence_as_list_of_gates(command_ids)

    def _record_sequence_as_list_of_command_ids(self, command_ids: list[int]):
        self._sequences_as_command_ids.append([])
        for command_id in command_ids:
            self._sequences_as_command_ids[-1].append(command_id)

    def _record_sequence_as_list_of_gates(self, command_ids: list[int]):
        self._sequences_as_gates.append([])
        for command_id in command_ids:
            command = self.command_registry.get_command_by_id(command_id)
            for gate in command:
                self._sequences_as_gates[-1].append(gate)

    def _serialize_sequences(self):
        result = ""
        for i, sequence in enumerate(self._sequences_as_gates):
            result += f"Sequence {i}:\n"
            result += f"\tCommand IDs: {self._sequences_as_command_ids[i]}\n"
            result += f"\tGates:\n"
            for j, operation in enumerate(sequence):
                result += f"\t\t{j}: {operation}\n"
            result += "\n"
        return result

    def verify_sequences(self):
        """
        Checks that the application of all gates in a sequence to the |00>
        state correctly recovers to the |00> state at the end.
        """
        for i, sequence in enumerate(self._sequences_as_gates):
            ground_state = np.kron(np.array([1, 0]), np.array([1, 0]))
            ground_state_rho = np.outer(ground_state, ground_state.conj())
            rho = ground_state_rho
            for gate in sequence:
                rho = gate.matrix() @ rho @ gate.matrix().conj().T

            assert np.allclose(rho, ground_state_rho), f"expected to recover to at {ground_state_rho}, got {rho}"

        print(f"Verification passed for all {len(self._sequences_as_gates)} sequence(s).")

    def print_sequences(self):
        print(self._serialize_sequences())

    def save_to_file(self, path: Union[str, Path]):
        with open(path, "w+") as f:
            f.write(self._serialize_sequences())
