from quam.core import quam_dataclass
from quam.components.channels import InOutIQChannel, InOutMWChannel
import numpy as np

__all__ = ["ReadoutResonator", "ReadoutResonatorIQ", "ReadoutResonatorMW"]


@quam_dataclass
class ReadoutResonatorBase:
    """QuAM component for a readout resonator

    Args:
        depletion_time (int): the resonator depletion time in ns.
        frequency_bare (int, float): the bare resonator frequency in Hz.
    """

    depletion_time: int = 5000
    frequency_bare: float = None

    f_01: float = None
    f_12: float = None
    confusion_matrix: list = None
    
    gef_centers : list = None
    gef_confusion_matrix : list = None


@quam_dataclass
class ReadoutResonatorIQ(InOutIQChannel, ReadoutResonatorBase):
    time_of_flight = 28  # smallest deviation from default (24ns) to work with Qualibrate

    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    # def get_output_power(self, operation, Z=50) -> float:
    #     power = self.frequency_converter_up.power
    #     amplitude = self.operations[operation].amplitude
    #     x_mw = 10 ** (power / 10)
    #     x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
    #     return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

@quam_dataclass
class ReadoutResonatorMW(InOutMWChannel, ReadoutResonatorBase):
    time_of_flight = 28

    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency

    def get_output_power(self, operation, Z=50) -> float:
        power = self.opx_output.full_scale_power_dbm
        amplitude = self.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)


ReadoutResonator = ReadoutResonatorIQ

