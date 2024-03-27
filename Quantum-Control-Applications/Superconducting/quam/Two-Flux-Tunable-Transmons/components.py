from dataclasses import field
from typing import List, Union

from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel
from quam.components.hardware import LocalOscillator, Mixer
from quam.core import QuamRoot, quam_dataclass

__all__ = ["Transmon", "QuAM"]


@quam_dataclass
class Transmon(QuamComponent):
    """Example QuAM component for a transmon qubit."""

    id: Union[int, str]

    xy: IQChannel = None
    z: SingleChannel = None

    resonator: InOutIQChannel = None

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    mixers: List[Mixer] = field(default_factory=list)
    qubits: List[Transmon] = field(default_factory=list)
    resonators: List[InOutIQChannel] = field(default_factory=list)
    local_oscillators: List[LocalOscillator] = field(default_factory=list)
    wiring: dict = field(default_factory=dict)
