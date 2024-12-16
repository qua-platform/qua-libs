from typing import Union

from quam.core import quam_dataclass
from quam import QuamComponent
from quam.components.channels import IQChannel, MWChannel
import numpy as np

__all__ = ["StarkInducedZZ", "StarkInducedZZIQ", "StarkInducedZZMW"]

@quam_dataclass
class StarkInducedZZBase:
    target_qubit_LO_frequency: int
    target_qubit_IF_frequency: int
    detuning: int

@quam_dataclass
class StarkInducedZZIQ(IQChannel, StarkInducedZZBase):

    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    @property
    def inferred_intermediate_frequency(self):
        return self.target_qubit_LO_frequency + self.target_qubit_IF_frequency - self.LO_frequency + self.detuning

    # def get_output_power(self, operation, Z=50) -> float:
    #     power = self.frequency_converter_up.power
    #     amplitude = self.operations[operation].amplitude
    #     x_mw = 10 ** (power / 10)
    #     x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
    #     return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

@quam_dataclass
class StarkInducedZZMW(MWChannel, StarkInducedZZBase):
    @property
    def inferred_intermediate_frequency(self):
        return self.target_qubit_LO_frequency + self.target_qubit_IF_frequency - self.LO_frequency + self.detuning

    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency

    # add property of upconverter here?
    def get_output_power(self, operation, Z=50) -> float:
        power = self.opx_output.full_scale_power_dbm
        amplitude = self.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)


StarkInducedZZ = StarkInducedZZIQ

