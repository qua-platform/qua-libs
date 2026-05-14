from dataclasses import field
from typing import Dict, Any, Union, Optional

from quam.core import quam_dataclass, QuamComponent
from quam.components.channels import MWChannel
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.architecture.superconducting.qpu import BaseQuam


@quam_dataclass
class StorageCavity(QuamComponent):
    """A storage cavity mode with a direct XY drive (MW-FEM) and no readout."""

    id: Union[int, str]
    xy: XYDriveMW = None
    f_01: float = None
    T1: float = None
    T2: float = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"cavity{self.id}"


@quam_dataclass
class ParametricDriveMW(MWChannel):
    """MW-FEM channel for parametric coupling tones at mode-difference frequencies.

    Sits on the flux line and carries AC tones that activate beam-splitter /
    two-mode-squeezing interactions between the transmon, readout resonator,
    and storage cavity.
    """

    transmon_frequency: float = None
    cavity_frequency: float = None
    readout_frequency: float = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def cavity_transmon_detuning(self) -> Optional[float]:
        if self.cavity_frequency is not None and self.transmon_frequency is not None:
            return abs(self.cavity_frequency - self.transmon_frequency)
        return None

    @property
    def cavity_readout_detuning(self) -> Optional[float]:
        if self.cavity_frequency is not None and self.readout_frequency is not None:
            return abs(self.cavity_frequency - self.readout_frequency)
        return None

    @property
    def readout_transmon_detuning(self) -> Optional[float]:
        if self.readout_frequency is not None and self.transmon_frequency is not None:
            return abs(self.readout_frequency - self.transmon_frequency)
        return None


@quam_dataclass
class ParametricCavityTransmon(FixedFrequencyTransmon):
    """Fixed-frequency transmon coupled to a readout resonator and a storage
    cavity via parametric drives on the flux line.

    Inherits ``xy`` (transmon XY drive) and ``resonator`` (readout) from
    ``FixedFrequencyTransmon``.  Adds the storage cavity with its own direct
    XY drive and a parametric drive channel for inter-mode coupling tones.
    """

    cavity: StorageCavity = None
    parametric_drive: ParametricDriveMW = None


@quam_dataclass
class Quam(BaseQuam):
    """QuAM root for a parametric-cavity device (MW-FEM only)."""

    qubits: Dict[str, ParametricCavityTransmon] = field(default_factory=dict)
