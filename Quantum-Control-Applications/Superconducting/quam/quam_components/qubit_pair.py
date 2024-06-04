from typing import Dict, Any
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler


__all__ = ["QubitPair"]


@quam_dataclass
class QubitPair(QuamComponent):
    qubit_control: Transmon
    qubit_target: Transmon
    coupler: TunableCoupler = None

    extras: Dict[str, Any] = field(default_factory=dict)
