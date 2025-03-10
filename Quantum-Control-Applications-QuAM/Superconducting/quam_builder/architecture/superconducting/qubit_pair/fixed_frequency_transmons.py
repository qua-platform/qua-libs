from typing import Dict, Any, Optional, Union
from dataclasses import field
from qm.qua import align, wait

from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.qubit.fixed_frequency_transmon import (
    FixedFrequencyTransmon,
)
from quam_builder.architecture.superconducting.components.cross_resonance import (
    CrossResonanceIQ,
    CrossResonanceMW,
)
from quam_builder.architecture.superconducting.components.zz_drive import (
    ZZDriveIQ,
    ZZDriveMW,
)


__all__ = ["FixedFrequencyTransmonPair"]


@quam_dataclass
class FixedFrequencyTransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: FixedFrequencyTransmon = None
    qubit_target: FixedFrequencyTransmon = None

    cross_resonance: Optional[Union[CrossResonanceMW, CrossResonanceIQ]] = None
    zz_drive: Optional[Union[ZZDriveMW, ZZDriveIQ]] = None
    confusion: list = None

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon pair"""
        return self.id if isinstance(self.id, str) else f"q{self.qubit_control.id}-{self.qubit_target.id}"

    def align(self):
        align(
            self.qubit_control.xy.name,
            self.qubit_control.resonator.name,
            self.qubit_target.xy.name,
            self.qubit_target.resonator.name,
        )

    def wait(self, duration):
        wait(
            duration,
            self.qubit_control.xy.name,
            self.qubit_control.resonator.name,
            self.qubit_target.xy.name,
            self.qubit_target.resonator.name,
        )
