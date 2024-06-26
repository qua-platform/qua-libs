from quam.core import quam_dataclass
from .single_channel_1k import SingleChannel1k

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(SingleChannel1k):
    # Optimize for DC pulses
    output_mode: str = "amplified"
    upsampling_mode: str = "pulse"
