from pathlib import Path
from typing import Union

from .gates import PhasedXZ, CZ, CNOT, Gate

Command = list[Gate]


class CommandRegistry:
    def __init__(self):
        self._current_command_id = 0
        self._commands: dict[int, Command] = {}

    def register_phase_xz(self, q, x, z, a):
        gate = PhasedXZ(q=q, z=z, x=x, a=a)
        self._register_gate(gate)

    def register_cz(self):
        gate = CZ()
        self._register_gate(gate)

    def register_cnot(self, q):
        gate = CNOT(q=q)
        self._register_gate(gate)

    def register_preparation(self):
        pass

    def register_measurement(self):
        pass

    def _register_gate(self, gate: Gate):
        if self._current_command_id in self._commands:
            command_list = self._commands[self._current_command_id]
        else:
            command_list = []
            self._commands[self._current_command_id] = command_list
        command_list.append(gate)

    def get_command_by_id(self, command_id: int):
        return self._commands[command_id]

    def set_current_command_id(self, command_id: int):
        self._current_command_id = command_id

    def _serialize_commands(self):
        result = ""
        for i in self._commands.keys():
            result += f"Command {i}:\n"
            for j, gate in enumerate(self._commands[i]):
                result += f"\t{j}: {gate}\n"
            result += "\n"
        return result

    def print_commands(self):
        print(self._serialize_commands())

    def save_to_file(self, path: Union[str, Path]):
        with open(path, 'w+') as f:
            f.write(self._serialize_commands())
