from typing import Dict, Any, Optional, Union
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler


__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: Transmon
    qubit_target: Transmon
    coupler: TunableCoupler = None
    coupler: Optional[TunableCoupler] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
