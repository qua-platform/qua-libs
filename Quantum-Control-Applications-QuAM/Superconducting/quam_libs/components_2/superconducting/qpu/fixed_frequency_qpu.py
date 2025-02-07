from quam.core import quam_dataclass
from ..qubit.fixed_frequency_transmon import FixedFrequencyTransmon
from ..qubit_pair.fixed_frequency_transmons import TransmonPair
from quam_libs.components.superconducting.qpu.base_quam import BaseQuAM

from dataclasses import field
from typing import Dict, ClassVar, Type

__all__ = ["QuAM", "FixedFrequencyTransmon", "TransmonPair"]



@quam_dataclass
class QuAM(BaseQuAM):
    """Example QuAM root component."""

    qubits: Dict[str, FixedFrequencyTransmon] = field(default_factory=dict)
    qubit_type: ClassVar[Type[FixedFrequencyTransmon]] = FixedFrequencyTransmon
    qubit_pairs: Dict[str, TransmonPair] = field(default_factory=dict)
    qubit_pair_type: ClassVar[Type[TransmonPair]] = TransmonPair

    def load(self, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)
