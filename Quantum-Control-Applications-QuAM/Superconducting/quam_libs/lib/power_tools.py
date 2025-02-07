import numpy as np

from typing import Optional

from quam.components import MWChannel


def set_output_power_mw_channel(channel: MWChannel, power_in_dbm: float, operation: str,
                     full_scale_power_dbm: Optional[int] = None, max_amplitude: Optional[float] = 1):
    """
    Sets the power level in dBm for a specified operation, increasing the full-scale power
    in 3 dB steps if necessary until it covers the target power level, then scaling the
    given operation’s amplitude to match exactly the target power level.

    Parameters:
        power_in_dbm (float): The target power level in dBm for the operation.
        full_scale_power_dbm (Optional[int]): The full-scale power in dBm within [-41, 10] in 3 dB increments.
        operation (Optional[str]): The operation for which the power setting is applied.

    """
    allowed_full_scale_power_in_dbm_values = np.arange(-41, 11, 3)

    if full_scale_power_dbm is not None:
        if full_scale_power_dbm < -20 or full_scale_power_dbm not in allowed_full_scale_power_in_dbm_values:
            raise ValueError(f"Expected full_scale_power_dbm to be > -20 in QOP3.2.0, or "
                             f"in range [-41, 10] in steps of 3 dB, got {full_scale_power_dbm}.")

        if power_in_dbm > full_scale_power_dbm:
            raise ValueError(f"Can't fix full_scale_power_dbm to {full_scale_power_dbm} dBm since it is "
                             f"less than the target power {power_in_dbm} dBm.")

        channel.opx_output.full_scale_power_dbm = full_scale_power_dbm

    if power_in_dbm > 10:
        raise ValueError(f"Expected `power_in_dbm` to be <10 dBm, got {power_in_dbm}")

    # use a temporary variable for node.record_state_updates
    temp_full_scale_power_dbm = channel.opx_output.full_scale_power_dbm

    while calculate_voltage_scaling_factor(
            fixed_power_dBm=temp_full_scale_power_dbm,
            target_power_dBm=power_in_dbm,
    ) > max_amplitude:
        temp_full_scale_power_dbm = temp_full_scale_power_dbm + 3

    if temp_full_scale_power_dbm not in allowed_full_scale_power_in_dbm_values:
        raise ValueError(f"Expected full_scale_power_dbm to be in range [-41, 10] "
                         f"in steps of 3 dB, got {temp_full_scale_power_dbm}.")

    channel.operations[operation].amplitude = calculate_voltage_scaling_factor(
        fixed_power_dBm=temp_full_scale_power_dbm,
        target_power_dBm=power_in_dbm
    )

    channel.opx_output.full_scale_power_dbm = temp_full_scale_power_dbm

    return {
        "full_scale_power_dbm": temp_full_scale_power_dbm,
        "amplitude": channel.operations[operation].amplitude
    }


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
