from quam.core import quam_dataclass
from ..qubit.fixed_frequency_transmon import Transmon
from ..qubit_pair.fixed_frequency_transmons import TransmonPair
from base_quam import Base_QuAM


from dataclasses import field
from typing import Dict

__all__ = ["QuAM"]



@quam_dataclass
class QuAM(Base_QuAM):
    """Example QuAM root component."""

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    qubit_pairs: Dict[str, TransmonPair] = field(default_factory=dict)

