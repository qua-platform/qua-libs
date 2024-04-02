from dataclasses import field
from typing import List, Union, Dict

from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass

from qm.qua import set_dc_offset

# import macros

__all__ = ["Transmon", "FluxLine", "ReadoutResonator", "QuAM"]


@quam_dataclass
class FluxLine(SingleChannel):
    """Example QuAM component for a transmon qubit."""

    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0

    def to_independent_idle(self):  # TODO: put the functions here
        set_dc_offset(self.name, "single", self.independent_offset)

    def to_joint_idle(self):
        set_dc_offset(self.name, "single", self.joint_offset)

    def to_min(self):
        set_dc_offset(self.name, "single", self.min_offset)


@quam_dataclass
class ReadoutResonator(InOutIQChannel):
    """ QuAM component for a readout resonator

    Args:
        depletion_time(int): the resonator depletion time in ns.
    """
    depletion_time: int = 1000


@quam_dataclass
class Transmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        thermalization_time (int): An integer.
        T1 (str): A string.
    """

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None

    resonator: ReadoutResonator = None

    T1: int = 10_000
    thermalization_time: int = "#./T1"

    # @property
    # def thermalization_time(self):
    #     return

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""
    octave: Octave = None

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    wiring: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)
    @property
    def active_qubits(self) -> List[Transmon]:
        return [self.qubits[q] for q in self.active_qubit_names]
