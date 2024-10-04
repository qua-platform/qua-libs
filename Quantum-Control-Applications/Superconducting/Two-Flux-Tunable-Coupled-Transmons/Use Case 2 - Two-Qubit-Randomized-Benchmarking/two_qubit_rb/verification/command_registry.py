from pathlib import Path
from typing import Union, Callable, Literal

from qualang_tools.bakery.bakery import Baking

from .gates import PhasedXZ, CZ, CNOT, Gate

Command = list[Gate]


class CommandRegistry:
    """
    Dataclass to track which single- or two-qubit gates are being baked in order
    to construct a two-qubit randomized benchmarking command.
    """

    def __init__(self):
        self._current_command_id = 0
        self._commands: dict[int, Command] = {}
        self._is_finished = False

    def register_phase_xz(self, q, x, z, a):
        gate = PhasedXZ(q=q, z=z, x=x, a=a)
        self._register_gate(gate)

    def register_cz(self):
        gate = CZ()
        self._register_gate(gate)

    def register_cnot(self, q):
        gate = CNOT(q=q)
        self._register_gate(gate)

    def _register_gate(self, gate: Gate):
        if self.is_finished():
            return
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
        with open(path, "w+") as f:
            f.write(self._serialize_commands())

    def finish(self):
        """disable the incidental recording of any more commands."""
        self._is_finished = True

    def is_finished(self):
        return self._is_finished


PhasedXZGeneratorFunc = Callable[[Baking, int, float, float, float], None]
SingleQubitGateGeneratorFunc = Union[PhasedXZGeneratorFunc]


def decorate_single_qubit_generator_with_command_recording(
    single_qubit_gate_generator: PhasedXZGeneratorFunc, command_registry: CommandRegistry
) -> PhasedXZGeneratorFunc:
    """
    Decorates the `single_qubit_gate_generator` function so that it records
    every function call an input parameters it receives and registers them
    with the provided `command_registry`.
    """

    def decorated_generator(baker: Baking, q: int, x: float, z: float, a: float):
        command_registry.register_phase_xz(q=q, x=x, z=z, a=a)
        return single_qubit_gate_generator(baker, q, x, z, a)

    return decorated_generator


CZGeneratorFunc = Callable[[Baking, int, int], None]
CNOTGeneratorFunc = Callable[[Baking, int, int], None]
TwoQubitGateGeneratorFunc = Union[CZGeneratorFunc, CNOTGeneratorFunc]
TwoQubitGateGenerator = dict[Literal["CZ", "CNOT"], TwoQubitGateGeneratorFunc]


def decorate_two_qubit_gate_generator_with_command_recording(
    two_qubit_gate_generator: TwoQubitGateGenerator, command_registry: CommandRegistry
) -> TwoQubitGateGenerator:
    """
    Decorates the `single_qubit_gate_generator` function so that it
    records every call to itself and registers the corresponding input parameters.
    """
    decorated_two_qubit_gate_generator = {}
    for gate_name, gate_generator in two_qubit_gate_generator.items():
        if gate_name == "CZ":

            def decorated_generator(*args):
                command_registry.register_cz()
                return gate_generator(*args)

        elif gate_name == "CNOT":

            def decorated_generator(*args):
                control_qubit_index = args[1]
                command_registry.register_cnot(q=control_qubit_index)
                return gate_generator(*args)

        else:
            raise NotImplementedError(
                f"Command recording not implemented for {gate_name}. Please contact customer success."
            )

        decorated_two_qubit_gate_generator[gate_name] = decorated_generator

    return decorated_two_qubit_gate_generator
