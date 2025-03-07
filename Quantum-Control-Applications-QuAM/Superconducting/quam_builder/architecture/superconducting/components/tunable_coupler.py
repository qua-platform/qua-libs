from quam.components import SingleChannel
from quam.core import quam_dataclass
from typing import Literal

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannel):
    """
    Example QuAM component for a tunable coupler.

    Attributes:
        decouple_offset (float): the coupler flux bias for which the interaction is off.
        interaction_offset (float): the coupler flux bias for which the interaction is ON.
        arbitrary_offset (float): an arbitrary coupler flux bias.
        flux_point (str): name of the flux point to set the coupler at. Can be among ["off", "on", "arbitrary", "zero"]. Default is "off".
        settle_time (float): the flux line settle time in ns. The value will be cast to an integer multiple of 4ns automatically.
    """

    decouple_offset: float = 0.0
    interaction_offset: float = 0.0
    arbitrary_offset: float = 0.0
    flux_point: Literal["off", "on", "arbitrary", "zero"] = "off"
    settle_time: float = None

    def settle(self):
        """Wait for the flux bias to settle"""
        if self.settle_time is not None:
            self.wait(int(self.settle_time) // 4 * 4)

    def to_decouple_idle(self):
        """Set the tunable coupler to the decouple offset."""
        self.set_dc_offset(self.decouple_offset)

    def to_interaction_idle(self):
        """Set the tunable coupler to the interaction offset."""
        self.set_dc_offset(self.interaction_offset)

    def to_arbitrary_idle(self):
        """Set the tunable coupler to the arbitrary offset."""
        self.set_dc_offset(self.arbitrary_offset)

    def to_zero(self):
        """Set the tunable coupler to 0V."""
        self.set_dc_offset(0.0)
