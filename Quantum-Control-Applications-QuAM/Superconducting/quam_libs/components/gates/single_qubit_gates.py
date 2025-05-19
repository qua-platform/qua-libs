from abc import ABC, abstractmethod
from typing import Any

from qm.qua import declare
from quam.components.macro import QubitMacro, PulseMacro
from quam.core import quam_dataclass, QuamComponent
import numpy as np

__all__ = ["SingleQubitGate", "SinglePulseGate", "VirtualZGate", "HadamardGate"]

from quam_libs.macros import active_reset, readout_state


@quam_dataclass
class SingleQubitGate(QuamComponent, ABC):
    @property
    def qubit(self):
        from ..transmon import Transmon

        if isinstance(self.parent, Transmon):
            return self.parent
        elif hasattr(self.parent, "parent") and isinstance(self.parent.parent, Transmon):
            return self.parent.parent
        else:
            raise AttributeError("SingleQubitGate is not attached to a qubit. 1Q_gate: {self}")

    def __call__(self):
        self.execute()

    @abstractmethod
    def execute(self, *args, **kwargs):  # TODO Accomodate differing arguments
        pass


@quam_dataclass
class SinglePulseGate(SingleQubitGate):
    """Single-qubit gate for a qubit consisting of a single pulse

    Args:
        pulse: Name of pulse to be played on qubit. Should be a key in
            `channel.operations` for one of the qubit's channels

    """

    pulse: str

    def execute(self, amplitude_scale=None, duration=None):
        self.qubit.play_pulse(self.pulse, amplitude_scale=amplitude_scale, duration=duration)


@quam_dataclass
class VirtualZGate(SingleQubitGate):
    """Single-qubit gate for a qubit consisting of a single pulse

    Args:
        angle: Angle of rotation in radians

    """

    angle: float = np.pi

    def execute(self, angle=None):
        if angle is None:
            angle = self.angle
        self.qubit.xy.frame_rotation(angle)


@quam_dataclass
class HadamardGate(QubitMacro):
    """single qubit Hadamard gate

    """

    def apply(self):
        self.qubit.xy.play('y90')
        self.qubit.xy.play('x180')
        # self.qubit.macros["Z"]()
        # self.qubit.macros["sY"]()


class SYGate(PulseMacro):
    def __init__(self, pulse):
        self.pulse = pulse

    def execute(self, amplitude_scale=None, duration=None):
        self.qubit.play_pulse(self.pulse)


class ResetGate(QubitMacro):

    def apply(self, *args, **kwargs) -> Any:
        self.qubit.wait(self.qubit.thermalization_time * 0.25)
        active_reset(self.qubit)


# def reset(q: int) -> None:
#     qubit = machine.qubits[index2qubit[q]]
#     qubit.wait(qubit.thermalization_time * 0.25)
#     qua.reset_frame(machine.qubits[index2qubit[q]].xy.name)

class MeasurementGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        state = declare(int)
        readout_state(self.qubit, state)
        return state


class XGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('x180')


# def x(q: int) -> None:
#     machine.qubits[index2qubit[q]].xy.play("x180")

class YGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('y180')


# def y(q: int) -> None:
#     machine.qubits[index2qubit[q]].xy.play("y180")

class SXGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('x90')


# def sx(q: int) -> None:
#     machine.qubits[index2qubit[q]].xy.play("x90")

class RZGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.frame_rotation(args[0])


# def rz(angle: QuaExpression, q: int) -> None:
#     machine.qubits[index2qubit[q]].xy.frame_rotation(angle)

# class CZGate(QubitMacro):
#     def apply(self, *args, **kwargs) -> Any:
#         self.qubit.xy.play('x90')
#
# def cz(control: int, target: int):
#     machine.active_qubit_pairs[
#         pair2index[f"{index2qubit[control]}-{index2qubit[target]}"]
#     ].gates["Cz"].execute()
#

# def cx(control:int, target:int):
#     qp = machine.active_qubit_pairs[pair2index[f"{index2qubit[control]}-{index2qubit[target]}"]]
#     qp.gates['CNotGate_TC']()

class DelayGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        duration = args[0]
        self.qubit.wait(duration)

# def delay(duration: QuaExpression, *qubits: int) -> None:
#     for q in qubits:
#         qua.wait(
#             duration,
#             f"{index2qubit[q]}$xy",
#             f"{index2qubit[q]}$rr",
#             f"{index2qubit[q]}$z",
#         )
