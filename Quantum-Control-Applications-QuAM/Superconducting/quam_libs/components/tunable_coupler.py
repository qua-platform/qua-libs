from quam.components import SingleChannel
from quam.components.ports import LFFEMAnalogOutputPort
from quam.core import quam_dataclass

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannel):
    """
    Example QuAM component for a tunable coupler.

    Args:
        decouple_offset (float): the tunable coupler for which the .
        interaction_offset (float): the tunable coupler for which the .
    """

    output_mode: str = "direct" # "amplified"
    upsampling_mode: str = "pulse"

    decouple_offset: float = 0.0
    interaction_offset: float = 0.0

    def __post_init__(self):
        if isinstance(self.opx_output, LFFEMAnalogOutputPort):
            self.opx_output.upsampling_mode = self.upsampling_mode
            self.opx_output.output_mode = self.output_mode

    def to_decouple_idle(self):
        """Set the tunable coupler to the decouple offset"""
        self.set_dc_offset(self.decouple_offset)

    def to_interaction_idle(self):
        """Set the tunable coupler to the interaction offset"""
        self.set_dc_offset(self.interaction_offset)
