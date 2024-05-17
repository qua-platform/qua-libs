from quam.core import quam_dataclass
from quam.components.channels import InOutIQChannel

__all__ = ["ReadoutResonator"]


@quam_dataclass
class ReadoutResonator(InOutIQChannel):
    """QuAM component for a readout resonator

    Args:
        depletion_time (int): the resonator depletion time in ns.
        frequency_bare (int, float): the bare resonator frequency in Hz.
    """

    depletion_time: int = 1000
    frequency_bare: float = None

    f_01: float = None
    f_12: float = None
