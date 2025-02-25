from typing import Optional

from quam.core import quam_dataclass
from quam.components.channels import InOutIQChannel, InOutMWChannel
from quam_libs.power_tools import calculate_voltage_scaling_factor, set_output_power_mw_channel
import numpy as np

from qualang_tools.units import unit

__all__ = ["ReadoutResonatorIQ", "ReadoutResonatorMW"]


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
    GEF_frequency_shift: float = None

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
        return calculate_voltage_scaling_factor(fixed_power_dBm, target_power_dBm)


@quam_dataclass
class ReadoutResonatorIQ(InOutIQChannel, ReadoutResonatorBase):
    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    def get_output_power(self, operation, Z=50) -> float:
        u = unit(coerce_to_integer=True)
        amplitude = self.operations[operation].amplitude
        return self.frequency_converter_up.gain + u.volts2dBm(amplitude, Z=Z)

    def set_output_power(
        self,
        power_in_dbm: float,
        gain: Optional[int] = None,
        max_amplitude: Optional[float] = None,
        Z: int = 50,
        operation: Optional[str] = "readout",
    ):
        """
        Configure the output power for a specific operation by setting the gain or amplitude.
        Note that exactly one of `gain` or `amplitude` must be specified and the function calculates
        the other parameter specifically to meet the desired output power.

        Parameters:
            power_in_dbm (float): Desired output power in dBm.
            gain (Optional[int]): Optional gain in dB to set, must be within [-20, 20].
            max_amplitude (Optional[float]): Optional pulse amplitude in volts, must be within [-0.5, 0.5).
            Z (int): Impedance in ohms, default is 50.
            operation (Optional[str]): Name of the operation to configure, default is "readout".

        Raises:
            RuntimeError: If neither nor both `gain` and `amplitude` are specified.
            ValueError: If `gain` or `amplitude` is outside their valid ranges.

        """

        u = unit(coerce_to_integer=True)

        if not ((max_amplitude is None) ^ (gain is None)):
            raise RuntimeError("Either or gain or amplitude must be specified.")
        elif max_amplitude is not None:
            gain = round((power_in_dbm - u.volts2dBm(max_amplitude, Z=Z)) * 2) / 2
            gain = min(max(gain, 20), -20)
            amplitude = u.dBm2volts(power_in_dbm - gain)
        elif gain is not None:
            amplitude = u.dBm2volts(power_in_dbm - self.frequency_converter_up.gain)

        if not -20 <= gain <= 20:
            raise ValueError(f"Expected Octave gain within [-20:0.5:20] dB, got {gain} dB.")

        if not -0.5 <= max_amplitude < 0.5:
            raise ValueError("The OPX+ pulse amplitude must be within [-0.5, 0.5) V.")

        print(f"Setting the Octave gain to {gain} dB")
        print(f"Setting the {operation} amplitude to {amplitude} V")

        self.frequency_converter_up.gain = gain
        self.operations[operation].amplitude = amplitude

        return {"gain": gain, "amplitude": max_amplitude}


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

    def set_output_power(
        self,
        power_in_dbm: float,
        full_scale_power_dbm: Optional[int] = None,
        max_amplitude: Optional[float] = 1,
        operation: Optional[str] = "readout",
    ):
        """
        Sets the power level in dBm for a specified operation, increasing the full-scale power
        in 3 dB steps if necessary until it covers the target power level, then scaling the
        given operation’s amplitude to match exactly the target power level.

        Parameters:
            power_in_dbm (float): The target power level in dBm for the operation.
            operation (str): The operation for which the power setting is applied.
            full_scale_power_dbm (Optional[int]): The full-scale power in dBm within [-41, 10] in 3 dB increments.
            max_amplitude (Optional[float]):
        """
        return set_output_power_mw_channel(self, power_in_dbm, operation, full_scale_power_dbm, max_amplitude)
