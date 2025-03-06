from quam.components import SingleChannel
from quam.core import quam_dataclass
from typing import Literal

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannel):
    """
    Example QuAM component for a tunable coupler.

    Args:
        decouple_offset (float): the coupler flux bias for which the interaction is off.
        interaction_offset (float): the coupler flux bias for which the interaction is ON.
    """

    output_mode: str = "direct"
    upsampling_mode: str = "pulse"

    decouple_offset: float = 0.0
    interaction_offset: float = 0.0
    arbitrary_offset: float = 0.0
    flux_point: Literal["off", "on", "arbitrary", "zero"] = "off"

    def to_decouple_idle(self):
        """Set the tunable coupler to the decouple offset"""
        self.set_dc_offset(self.decouple_offset)

    def to_interaction_idle(self):
        """Set the tunable coupler to the interaction offset"""
        self.set_dc_offset(self.interaction_offset)

    def to_arbitrary_idle(self):
        """Set the tunable coupler to the arbitrary offset"""
        self.set_dc_offset(self.arbitrary_offset)
