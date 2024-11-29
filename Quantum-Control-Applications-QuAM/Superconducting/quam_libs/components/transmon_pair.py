from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler
from .gates.two_qubit_gates import TwoQubitGate
from qm.qua import align

__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: Transmon = None
    qubit_target: Transmon = None
    coupler: Optional[TunableCoupler] = None
    gates: Dict[str, TwoQubitGate] = field(default_factory=dict)
    J2: float = 0
    detuning: float = 0
    confusion: Optional[List[List[float]]] = None
    mutual_flux_bias: List[float] = field(default_factory=lambda: [0, 0])
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon pair"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
    
    def align(self):
        if self.coupler:
            align(self.qubit_control.xy.name, self.qubit_control.z.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, 
                  self.qubit_target.z.name, self.qubit_target.resonator.name, self.coupler.name)
        else:
            align(self.qubit_control.xy.name, self.qubit_control.z.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, 
                  self.qubit_target.z.name, self.qubit_target.resonator.name)
            
    def to_mutual_idle(self):
        """Set the flux bias to the mutual idle offset"""
        self.qubit_control.z.set_dc_offset(self.mutual_flux_bias[0])
        self.qubit_target.z.set_dc_offset(self.mutual_flux_bias[1])
            
