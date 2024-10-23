from quam.components import SingleChannel
from quam.components.ports import LFFEMAnalogOutputPort
from quam.core import quam_dataclass


__all__ = ["FluxLine"]


@quam_dataclass
class FluxLine(SingleChannel):
    """QuAM component for a flux line.

    Args:
        independent_offset (float): the flux bias corresponding to the resonator maximum frequency when the active qubits are not interacting (min offset) in V.
        joint_offset (float): the flux bias corresponding to the resonator maximum frequency when the active qubits are interacting (joint offset) in V.
        min_offset (float): the flux bias corresponding to the resonator minimum frequency in V.
    """
    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0
    arbitrary_offset: float = 0.0

    def to_independent_idle(self):
        """Set the flux bias to the independent offset"""
        self.set_dc_offset(self.independent_offset)

    def to_joint_idle(self):
        """Set the flux bias to the joint offset"""
        self.set_dc_offset(self.joint_offset)

    def to_min(self):
        """Set the flux bias to the min offset"""
        self.set_dc_offset(self.min_offset)

    def to_zero(self):
        """Set the flux bias to 0.0 V"""
        self.set_dc_offset(0.0)
