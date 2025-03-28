import numpy as np

from typing import Optional

from quam.components import IQChannel
from quam.components.channels import MWChannel

from qualang_tools.units import unit


def set_output_power_mw_channel(
    channel: MWChannel,
    power_in_dbm: float,
    operation: str,
    full_scale_power_dbm: Optional[int] = None,
    max_amplitude: Optional[float] = 1,
):
    """
    Sets the power level in dBm for a specified operation, increasing the full-scale power
    in 3 dB steps if necessary until it covers the target power level, then scaling the
    given operation’s amplitude to match exactly the target power level.

    Parameters:
        channel: A MW-FEM channel.
        power_in_dbm (float): The target power level in dBm for the operation.
        operation (str): The operation for which the power setting is applied.
        full_scale_power_dbm (Optional[int]): The full-scale power in dBm within [-41, 10] in 3 dB increments.
        max_amplitude (Optional[float]):

    """
    allowed_full_scale_power_in_dbm_values = np.arange(-11, 17, 3)

    if full_scale_power_dbm is not None:
        if full_scale_power_dbm < -20 or full_scale_power_dbm not in allowed_full_scale_power_in_dbm_values:
            raise ValueError(
                f"Expected full_scale_power_dbm to be > -20 in QOP3.2.0, or "
                f"in range [-41, 10] in steps of 3 dB, got {full_scale_power_dbm}."
            )

        if power_in_dbm > full_scale_power_dbm:
            raise ValueError(
                f"Can't fix full_scale_power_dbm to {full_scale_power_dbm} dBm since it is "
                f"less than the target power {power_in_dbm} dBm."
            )

        channel.opx_output.full_scale_power_dbm = full_scale_power_dbm

    if power_in_dbm > 10:
        raise ValueError(f"Expected `power_in_dbm` to be <10 dBm, got {power_in_dbm}")

    # use a temporary variable for node.record_state_updates
    temp_full_scale_power_dbm = channel.opx_output.full_scale_power_dbm

    while (
        calculate_voltage_scaling_factor(
            fixed_power_dBm=temp_full_scale_power_dbm,
            target_power_dBm=power_in_dbm,
        )
        > max_amplitude
    ):
        temp_full_scale_power_dbm = temp_full_scale_power_dbm + 3

    if temp_full_scale_power_dbm not in allowed_full_scale_power_in_dbm_values:
        raise ValueError(
            f"Expected full_scale_power_dbm to be in range [-41, 10] "
            f"in steps of 3 dB, got {temp_full_scale_power_dbm}."
        )

    channel.operations[operation].amplitude = calculate_voltage_scaling_factor(
        fixed_power_dBm=temp_full_scale_power_dbm, target_power_dBm=power_in_dbm
    )

    channel.opx_output.full_scale_power_dbm = temp_full_scale_power_dbm

    return {
        "full_scale_power_dbm": temp_full_scale_power_dbm,
        "amplitude": channel.operations[operation].amplitude,
    }


def get_output_power_mw_channel(channel: MWChannel, operation, Z=50) -> float:
    """
    Calculate the output power in dBm for a given MW-FEM channel.

    Parameters:
        channel: A MW-FEM channel.
        operation (str): The name of the operation to retrieve the amplitude.
        Z (float): The impedance in ohms. Default is 50 ohms.

    Returns:
        float: The output power in dBm.

    The function calculates the output power based on the full-scale power in dBm and the amplitude of the specified operation.
    """
    power = channel.opx_output.full_scale_power_dbm
    amplitude = channel.operations[operation].amplitude
    x_mw = 10 ** (power / 10)
    x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
    return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)


def set_output_power_iq_channel(
    channel: IQChannel,
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
        channel (IQChannel): The IQ channel.
        power_in_dbm (float): Desired output power in dBm.
        gain (Optional[int]): Optional Octave gain in dB to set, must be within [-20, 20].
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
        gain = max(min(gain, 20), -20)
        amplitude = u.dBm2volts(power_in_dbm - gain)
    elif gain is not None:
        amplitude = u.dBm2volts(power_in_dbm - channel.frequency_converter_up.gain)

    if not -20 <= gain <= 20:
        raise ValueError(f"Expected Octave gain within [-20:0.5:20] dB, got {gain} dB.")

    if not -0.5 <= max_amplitude < 0.5:
        raise ValueError("The OPX+ pulse amplitude must be within [-0.5, 0.5) V.")

    print(f"Setting the Octave gain to {gain} dB")
    print(f"Setting the {operation} amplitude to {amplitude} V")

    channel.frequency_converter_up.gain = gain
    channel.operations[operation].amplitude = amplitude

    return {"gain": gain, "amplitude": max_amplitude}


def get_output_power_iq_channel(channel: IQChannel, operation, Z=50) -> float:
    """
    Calculate the output power in dBm for a given IQ channel.

    Parameters:
        channel (IQChannel): The IQ channel.
        operation (str): The name of the operation to retrieve the amplitude.
        Z (float): The impedance in ohms. Default is 50 ohms.

    Returns:
        float: The output power in dBm.

    The function calculates the output power based on the amplitude of the specified operation and the gain of the frequency up-converter.
    It converts the amplitude to dBm using the specified impedance.
    """
    u = unit(coerce_to_integer=True)
    amplitude = channel.operations[operation].amplitude
    return channel.frequency_converter_up.gain + u.volts2dBm(amplitude, Z=Z)


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
