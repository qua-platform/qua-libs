from typing import Optional

from quam.core import quam_dataclass
from quam.components.channels import InOutIQChannel, InOutMWChannel
import numpy as np
from sympy.codegen.cnodes import static

from qualang_tools.units import unit

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

    gef_centers: list = None
    gef_confusion_matrix: list = None


@quam_dataclass
class ReadoutResonatorIQ(InOutIQChannel, ReadoutResonatorBase):
    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    def get_output_power(self, operation, Z=50) -> float:
        u = unit(coerce_to_integer=True)
        amplitude = self.operations[operation].amplitude
        return self.frequency_converter_up.gain + u.volts2dBm(amplitude, Z=Z)


@quam_dataclass
class ReadoutResonatorMW(InOutMWChannel, ReadoutResonatorBase):
    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency

    def get_output_power(self, operation, Z=50) -> float:
        power = self.opx_output.full_scale_power_dbm
        amplitude = self.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)


    def set_output_power(self, power_in_dbm: float, full_scale_power_dbm: Optional[int] = None,
                         operation: Optional[str] = 'readout'):
        """
        Sets the power level in dBm for a specified operation, increasing the full-scale power
        in 3 dB steps if necessary until it covers the target power level.

        Parameters:
        ----------
        power_in_dbm : float
            The target power level in dBm for the operation.

        full_scale_power_dbm : Optional[int], default=None
            The full-scale power limit in dBm within range [-41, 10] in 3 dB increments.

        operation : Optional[str], default='readout'
            The operation for which the power setting is applied. This operation’s
            amplitude is adjusted based on the calculated voltage scaling factor to
            match the target power level.

        """
        allowed_full_scale_power_in_dbm_values = np.arange(-41, 10, 3)

        if full_scale_power_dbm is not None:
            if full_scale_power_dbm < -20 or full_scale_power_dbm not in allowed_full_scale_power_in_dbm_values:
                raise ValueError(f"Expected full_scale_power_dbm to be > -20 in QOP3.2.0, or "
                                 f"in range [-41, 10] in steps of 3 dB, got {full_scale_power_dbm}.")

            if power_in_dbm > full_scale_power_dbm:
                raise ValueError(f"Can't fix full_scale_power_dbm to {full_scale_power_dbm} dBm since it is "
                                 f"less than the target power {power_in_dbm} dBm.")

            self.opx_output.full_scale_power_dbm = full_scale_power_dbm

        if power_in_dbm > 10:
            raise ValueError(f"Expected `power_in_dbm` to be <10 dBm, got {power_in_dbm}")

        while power_in_dbm > self.opx_output.full_scale_power_dbm:
            self.opx_output.full_scale_power_dbm += 3

        self.operations[operation].amplitude = self.calculate_voltage_scaling_factor(
            fixed_power_dBm=self.opx_output.full_scale_power_dbm,
            target_power_dBm=power_in_dbm
        )


    @staticmethod
    def calculate_voltage_scaling_factor(fixed_power_dBm: float, target_power_dBm: float):
        """
        Calculate the voltage scaling factor required to scale fixed power to target power.

        Parameters:
        fixed_power_dBm (float): The fixed power in dBm.
        target_power_dBm (float): The target power in dBm.

        Returns:
        float: The voltage scaling factor.
        """
        power_difference = target_power_dBm - fixed_power_dBm
        voltage_scaling_factor = 10 ** (power_difference / 20)
        return voltage_scaling_factor


ReadoutResonator = ReadoutResonatorIQ
