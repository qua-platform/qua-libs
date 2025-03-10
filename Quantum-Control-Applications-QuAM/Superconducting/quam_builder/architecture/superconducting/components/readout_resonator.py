from typing import Optional

from quam.core import quam_dataclass
from quam.components.channels import InOutIQChannel, InOutMWChannel
from quam_libs.power_tools import (
    calculate_voltage_scaling_factor,
    set_output_power_mw_channel,
    get_output_power_mw_channel,
    set_output_power_iq_channel,
    get_output_power_iq_channel,
)


__all__ = ["ReadoutResonatorIQ", "ReadoutResonatorMW"]


@quam_dataclass
class ReadoutResonatorBase:
    """
    QuAM component for a readout resonator.

    Attributes:
        depletion_time (int): The resonator depletion time in ns. Default is 16ns.
        frequency_bare (float): The bare resonator frequency in Hz.
        f_01 (float): The frequency of the 0-1 transition in Hz.
        f_12 (float): The frequency of the 1-2 transition in Hz.
        confusion_matrix (list): The confusion matrix for the resonator.
        gef_centers (list): The centers of the GEF states.
        gef_confusion_matrix (list): The confusion matrix for the GEF states.
        GEF_frequency_shift (float): The frequency shift for the GEF states.

    Methods:
        calculate_voltage_scaling_factor(fixed_power_dBm, target_power_dBm): Calculate the voltage scaling factor required to scale fixed power to target power.
    """

    depletion_time: int = 16
    frequency_bare: float = None

    f_01: float = None
    f_12: float = None
    confusion_matrix: list = None

    gef_centers: list = None
    gef_confusion_matrix: list = None
    GEF_frequency_shift: float = None

    @staticmethod
    def calculate_voltage_scaling_factor(
        fixed_power_dBm: float, target_power_dBm: float
    ):
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
        """Returns the up-converter/LO frequency in Hz."""
        return self.LO_frequency

    def get_output_power(self, operation, Z=50) -> float:
        """
        Calculate the output power in dBm of the specified operation.

        Parameters:
            operation (str): The name of the operation to retrieve the amplitude.
            Z (float): The impedance in ohms. Default is 50 ohms.

        Returns:
            float: The output power in dBm.

        The function calculates the output power based on the amplitude of the specified operation and the gain of the frequency up-converter.
        It converts the amplitude to dBm using the specified impedance.
        """
        return get_output_power_iq_channel(self, operation, Z)

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
        return set_output_power_iq_channel(
            self, power_in_dbm, gain, max_amplitude, Z, operation
        )


@quam_dataclass
class ReadoutResonatorMW(InOutMWChannel, ReadoutResonatorBase):
    @property
    def upconverter_frequency(self):
        """Returns the up-converter/LO frequency in Hz."""
        return self.opx_output.upconverter_frequency

    def get_output_power(self, operation, Z=50) -> float:
        """
        Calculate the output power in dBm of the specified operation.

        Parameters:
            operation (str): The name of the operation to retrieve the amplitude.
            Z (float): The impedance in ohms. Default is 50 ohms.

        Returns:
            float: The output power in dBm.

        The function calculates the output power based on the full-scale power in dBm and the amplitude of the specified operation.
        """
        return get_output_power_mw_channel(self, operation, Z)

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
        return set_output_power_mw_channel(
            self, power_in_dbm, operation, full_scale_power_dbm, max_amplitude
        )
