from typing import Dict, Any, Optional, Union, List
from dataclasses import field
from qm.qua import align, wait

from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import (
    FluxTunableTransmon,
)
from quam_builder.architecture.superconducting.components.tunable_coupler import (
    TunableCoupler,
)


__all__ = ["FluxTunableTransmonPair"]


@quam_dataclass
class FluxTunableTransmonPair(QuamComponent):
    id: Union[int, str]
    qubit_control: FluxTunableTransmon = None
    qubit_target: FluxTunableTransmon = None
    coupler: Optional[TunableCoupler] = None
    mutual_flux_bias: List[float] = field(default_factory=lambda: [0, 0])
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon pair"""
        return self.id if isinstance(self.id, str) else f"q{self.qubit_control.id}-{self.qubit_target.id}"

    def align(self):
        channels = [
            self.qubit_control.xy.name,
            self.qubit_control.z.name,
            self.qubit_control.resonator.name,
            self.qubit_target.xy.name,
            self.qubit_target.z.name,
            self.qubit_target.resonator.name,
        ]

        if self.coupler:
            channels += [self.coupler.name]

        if "Cz" in self.gates:
            if hasattr(self.gates["Cz"], "compensations"):
                for compensation in self.gates["Cz"].compensations:
                    channels += [
                        compensation["qubit"].xy.name,
                        compensation["qubit"].z.name,
                        compensation["qubit"].resonator.name,
                    ]

        align(*channels)

    def wait(self, duration):
        channels = [
            self.qubit_control.xy.name,
            self.qubit_control.z.name,
            self.qubit_control.resonator.name,
            self.qubit_target.xy.name,
            self.qubit_target.z.name,
            self.qubit_target.resonator.name,
        ]

        if self.coupler:
            channels += [self.coupler.name]

        if "Cz" in self.gates:
            if hasattr(self.gates["Cz"], "compensations"):
                for compensation in self.gates["Cz"].compensations:
                    channels += [
                        compensation["qubit"].xy.name,
                        compensation["qubit"].z.name,
                        compensation["qubit"].resonator.name,
                    ]

        wait(duration, *channels)

    def to_mutual_idle(self):
        """Set the flux bias to the mutual idle offset"""
        self.qubit_control.z.set_dc_offset(self.mutual_flux_bias[0])
        self.qubit_target.z.set_dc_offset(self.mutual_flux_bias[1])
