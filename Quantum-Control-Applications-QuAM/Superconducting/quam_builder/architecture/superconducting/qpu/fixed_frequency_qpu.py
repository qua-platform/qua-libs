from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon
from quam_builder.architecture.superconducting.qubit_pair import (
    FixedFrequencyTransmonPair,
)
from quam_builder.architecture.superconducting.qpu.base_quam import BaseQuAM

from dataclasses import field
from typing import Dict, ClassVar, Type

__all__ = ["QuAM", "FixedFrequencyTransmon", "FixedFrequencyTransmonPair"]


@quam_dataclass
class QuAM(BaseQuAM):
    """Example QuAM root component."""

    qubit_type: ClassVar[Type[FixedFrequencyTransmon]] = FixedFrequencyTransmon
    qubit_pair_type: ClassVar[Type[FixedFrequencyTransmonPair]] = (
        FixedFrequencyTransmonPair
    )

    qubits: Dict[str, FixedFrequencyTransmon] = field(default_factory=dict)
    qubit_pairs: Dict[str, FixedFrequencyTransmonPair] = field(default_factory=dict)

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)
