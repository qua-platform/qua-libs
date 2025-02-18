import warnings
from quam.core import quam_dataclass
from quam_builder.components.superconducting.qubit.flux_tunable_transmon import FluxTunableTransmon
from quam_builder.components.superconducting.qubit_pair.flux_tunable_transmons import TransmonPair
from quam_builder.components.superconducting.qpu.base_quam import BaseQuAM



from dataclasses import field
from typing import Dict, Union, ClassVar, Type

__all__ = ["QuAM", "FluxTunableTransmon", "TransmonPair"]


@quam_dataclass
class QuAM(BaseQuAM):
    """Example QuAM root component."""

    qubit_type: ClassVar[Type[FluxTunableTransmon]] = FluxTunableTransmon
    qubit_pair_type: ClassVar[Type[TransmonPair]] = TransmonPair

    qubits: Dict[str, FluxTunableTransmon] = field(default_factory=dict)
    qubit_pairs: Dict[str, TransmonPair] = field(default_factory=dict)

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)

    def apply_all_couplers_to_min(self) -> None:
        """Apply the offsets that bring all the active qubit pairs to a decoupled point."""
        for qp in self.active_qubit_pairs:
            if qp.coupler is not None:
                qp.coupler.to_decouple_idle()

    def apply_all_flux_to_joint_idle(self) -> None:
        """Apply the offsets that bring all the active qubits to the joint sweet spot."""
        for q in self.active_qubits:
            if q.z is not None:
                q.z.to_joint_idle()
            else:
                warnings.warn(f"Didn't find z-element on qubit {q.name}, didn't set to joint-idle")
        for q in self.qubits:
            if self.qubits[q] not in self.active_qubits:
                if self.qubits[q].z is not None:
                    self.qubits[q].z.to_min()
                else:
                    warnings.warn(f"Didn't find z-element on qubit {q}, didn't set to min")
        self.apply_all_couplers_to_min()

    def apply_all_flux_to_min(self) -> None:
        """Apply the offsets that bring all the active qubits to the minimum frequency point."""
        for q in self.qubits:
            if self.qubits[q].z is not None:
                self.qubits[q].z.to_min()
            else:
                warnings.warn(f"Didn't find z-element on qubit {q}, didn't set to min")
        self.apply_all_couplers_to_min()

    def apply_all_flux_to_zero(self) -> None:
        """Apply the offsets that bring all the active qubits to the zero bias point."""
        for q in self.active_qubits:
            q.z.to_zero()

    def set_all_fluxes(self, flux_point: str, target: Union[FluxTunableTransmon, TransmonPair]):
        if flux_point == "independent":
            assert isinstance(target, FluxTunableTransmon), "Independent flux point is only supported for individual transmons"
        elif flux_point == "pairwise":
            assert isinstance(target, TransmonPair), "Pairwise flux point is only supported for transmon pairs"

        target_bias = None
        if flux_point == "joint":
            self.apply_all_flux_to_joint_idle()
            if isinstance(target, TransmonPair):
                target_bias = target.mutual_flux_bias
            else:
                target_bias = target.z.joint_offset
        else:
            self.apply_all_flux_to_min()

        if flux_point == "independent":
            target.z.to_independent_idle()
            target_bias = target.z.independent_offset

        elif flux_point == "pairwise":
            target.to_mutual_idle()
            target_bias = target.mutual_flux_bias

        target.z.settle()
        target.align()
        return target_bias

