from typing import Dict, Any, Optional, Union
from dataclasses import field
from qm.qua import align, wait

from quam.core import QuamComponent, quam_dataclass
from ..qubit.flux_tunable_transmon import Transmon
from ..architectural_elements.cross_resonance import CrossResonance
from ..architectural_elements.zz_drive import ZZDrive


__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: Transmon = None
    qubit_target: Transmon = None
    cross_resonance: Optional[CrossResonance] = None
    zz_drive: Optional[ZZDrive] = None
    confusion: list = None

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def align(self):
        if (self.qubit_control.z is not None) and (self.qubit_target.z is not None):
            align(self.qubit_control.xy.name, self.qubit_control.z.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, self.qubit_target.z.name, self.qubit_target.resonator.name)
        else:
            align(self.qubit_control.xy.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, self.qubit_target.resonator.name)

    def wait(self, duration):
        if (self.qubit_control.z is not None) and (self.qubit_target.z is not None):
            wait(duration, self.qubit_control.xy.name, self.qubit_control.z.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, self.qubit_target.z.name, self.qubit_target.resonator.name)
        else:
            wait(duration, self.qubit_control.xy.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, self.qubit_target.resonator.name)
