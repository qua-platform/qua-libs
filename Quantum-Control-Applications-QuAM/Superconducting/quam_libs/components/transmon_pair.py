from typing import Dict, Any, Optional, Union
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler
from .cross_resonance import CrossResonance
from .zz_drive import ZZDrive


__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: Transmon = None
    qubit_target: Transmon = None
    coupler: Optional[TunableCoupler] = None
    cross_resonance: Optional[CrossResonance] = None
    zz_drive: Optional[ZZDrive] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
