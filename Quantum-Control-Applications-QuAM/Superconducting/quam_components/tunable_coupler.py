from quam.core import quam_dataclass
from .single_channel_lf_fem import SingleChannelLfFem

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannelLfFem):
    """
    Example QuAM component for a tunable coupler.
    """
    output_mode: str = "amplified"
    upsampling_mode: str = "pulse"
