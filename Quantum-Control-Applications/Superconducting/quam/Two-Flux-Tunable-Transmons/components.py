from dataclasses import field
from typing import List, Union

from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel
from quam.components.hardware import LocalOscillator, Mixer
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass

import macros

__all__ = ["Transmon", "FluxLine", "ReadoutResonator", "QuAM"]


@quam_dataclass
class FluxLine(SingleChannel):
    """Example QuAM component for a transmon qubit."""

    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0

    def to_independent_idle(self):
        macros.to_independent_idle(self)

    def to_joint_idle(self):
        macros.to_joint_idle(self)

    def to_min(self):
        macros.apply_z_to_min(self)

@quam_dataclass
class ReadoutResonator(InOutIQChannel):
    """Example QuAM component for a transmon qubit."""

    depletion_time: int = 1000


@quam_dataclass
class Transmon(QuamComponent):
    """Example QuAM component for a transmon qubit."""

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None

    resonator: ReadoutResonator = None

    T1: int = 10_000
    thermalization_time: int = "#./T1"

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""
    octave: Octave = None
    mixers: List[Mixer] = field(default_factory=list)
    qubits: List[Transmon] = field(default_factory=list)
    resonators: List[InOutIQChannel] = field(default_factory=list)
    local_oscillators: List[LocalOscillator] = field(default_factory=list)
    wiring: dict = field(default_factory=dict)

    active_qubits = "#./qubits"
    # @property
    # def active_qubits(self) -> List[qubits]:
    #     if self.architecture.active_qubits is None:
    #         return list(self.qubits.values())
    #     else:
    #         active_qubits = set(self.architecture.active_qubits)
    #         return [q for name, q in self.qubits.items() if name in active_qubits]
