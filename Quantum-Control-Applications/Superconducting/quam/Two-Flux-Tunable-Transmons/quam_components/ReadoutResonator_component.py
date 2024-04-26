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
    # TODO: update with inferred frequencies
    depletion_time: int = 1000
    frequency_bare: float = None

    @property
    def f_01(self):
        """The optimal frequency for discriminating the qubit between |0> and |1> (|g> -> |e>) in Hz"""
        return self.frequency_converter_up.LO_frequency + self.intermediate_frequency

    @property
    def f_12(self):
        """The optimal frequency for discriminating the qubit between |1> and |2> (|e> -> |f>) in Hz"""
        return self.frequency_converter_up.LO_frequency + self.intermediate_frequency
