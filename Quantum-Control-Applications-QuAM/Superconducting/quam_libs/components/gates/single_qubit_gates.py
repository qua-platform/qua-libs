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


class SYGate(PulseMacro):
    def __init__(self, pulse):
        self.pulse = pulse

    def execute(self, amplitude_scale=None, duration=None):
        self.qubit.play_pulse(self.pulse)


class ResetGate(QubitMacro):

    def apply(self, *args, **kwargs) -> Any:
        self.qubit.wait(self.qubit.thermalization_time * 0.25)
        active_reset(self.qubit)


class MeasurementGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        state = declare(int)
        readout_state(self.qubit, state)
        return state


class XGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('x180')


class YGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('y180')


class SXGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.play('x90')


class RZGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        self.qubit.xy.frame_rotation(args[0])


class DelayGate(QubitMacro):
    def apply(self, *args, **kwargs) -> Any:
        duration = args[0]
        self.qubit.wait(duration)


class UGate(QubitMacro):
    def _rz(self, angle):
        self.qubit.xy.frame_rotation(angle)

    def _x90(self):
        self.qubit.xy.play('x90')

    # implementation based on https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.U3Gate
    def apply(self, *args, **kwargs) -> Any:
        theta = args[0]
        phi = args[1]
        lambda_ = args[2]
        self._rz(lambda_)
        self._x90()
        self._rz(theta + np.pi)
        self._x90()
        self._rz(phi + np.pi)


class U1Gate(UGate):
    def apply(self, *args, **kwargs) -> Any:
        return super().apply(0, 0, args[0])


class U3Gate(UGate):
    pass
