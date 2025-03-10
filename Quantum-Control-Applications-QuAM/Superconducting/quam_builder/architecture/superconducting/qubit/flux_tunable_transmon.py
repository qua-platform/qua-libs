from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit.fixed_frequency_transmon import (
    FixedFrequencyTransmon,
)
from quam_builder.architecture.superconducting.components.flux_line import FluxLine
from qm.qua import align, wait

__all__ = ["FluxTunableTransmon"]


@quam_dataclass
class FluxTunableTransmon(FixedFrequencyTransmon):
    """
    Example QuAM component for a flux tunable transmon qubit.

    Args:
        z (FluxLine): The z drive component.
        resonator (ReadoutResonator): The readout resonator component.
        freq_vs_flux_01_quad_term (float):
        arbitrary_intermediate_frequency (float):
        phi0_current (float):
        phi0_voltage (float):
    """

    z: FluxLine = None
    freq_vs_flux_01_quad_term: float = 0.0
    arbitrary_intermediate_frequency: float = 0.0
    phi0_current: float = 0.0
    phi0_voltage: float = 0.0

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def __matmul__(self, other):
        if not isinstance(other, FluxTunableTransmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, " f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError("Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}")

        for qubit_pair in self._root.qubit_pairs.values():
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError("Qubit pair not found: qubit_control={self.name}, " "qubit_target={other.name}")

    def align(self):
        align(self.xy.name, self.z.name, self.resonator.name)

    def wait(self, duration):
        wait(duration, self.xy.name, self.z.name, self.resonator.name)
