from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

from .wiring_spec_enums import WiringLineType
from ..instruments.instrument_channels import InstrumentChannel


@dataclass(frozen=True)
class QubitReference:
    index: int


@dataclass(frozen=True)
class QubitPairReference:
    control_index: int
    target_index: int


ElementId = Union[QubitReference, QubitPairReference]


@dataclass
class Element:
    id: ElementId
    channels: Dict[WiringLineType, List[InstrumentChannel]] = field(default_factory=dict)

    def __str__(self):
        return str(self.channels)

    def __eq__(self, other):
        return self.id == other.id
