from quam.core import quam_dataclass
from quam.components.channels import IQChannel, MWChannel
from quam_builder.architecture.superconducting.qubit.base_transmon import BaseTransmon
from qm.qua import align, wait
from typing import Union

__all__ = ["FixedFrequencyTransmon"]


# todo: shall this on be the base Transmon directly?
@quam_dataclass
class FixedFrequencyTransmon(BaseTransmon):
    """
    Example QuAM component for a transmon qubit.

    Args:

    """

    xy_detuned: Union[MWChannel, IQChannel] = (
        None  # TODO: should probably belong into qubit pairs since it is for ZZ
    )

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def __matmul__(self, other):
        if not isinstance(other, FixedFrequencyTransmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, "
                f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError(
                "Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}"
            )

        for qubit_pair in self._root.qubit_pairs.values():
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError(
                "Qubit pair not found: qubit_control={self.name}, "
                "qubit_target={other.name}"
            )

    def align(self):
        align(self.xy.name, self.resonator.name)

    def wait(self, duration):
        wait(duration, self.xy.name, self.resonator.name)
