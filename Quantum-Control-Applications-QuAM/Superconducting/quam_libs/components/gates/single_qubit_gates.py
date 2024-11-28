from abc import ABC, abstractmethod
from quam.core import quam_dataclass, QuamComponent
import numpy as np

__all__ = ["SingleQubitGate", "SinglePulseGate", "VirtualZGate", "HadamardGate"]


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

    angle: float

    def execute(self, angle=None):
        if angle is None:
            angle = self.angle
        self.qubit.xy.frame_rotation(angle)

@quam_dataclass
class HadamardGate(SingleQubitGate):
    """single qubit Hadamard gate

    """
    def execute(self):
        self.qubit.gates["Z"]()
        self.qubit.gates["sY"]()
        
