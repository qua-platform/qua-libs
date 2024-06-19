from typing import Dict, Any, Optional
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler


__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QuamComponent):
    qubit_control: Transmon
    qubit_target: Transmon
    coupler: Optional[TunableCoupler] = None

    extras: Dict[str, Any] = field(default_factory=dict)
