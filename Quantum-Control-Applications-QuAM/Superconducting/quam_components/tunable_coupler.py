from quam.core import quam_dataclass
from .single_channel_lf_fem import SingleChannelLfFem

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannelLfFem):
    # Optimize for DC pulses
    output_mode: str = "amplified"
    upsampling_mode: str = "pulse"
