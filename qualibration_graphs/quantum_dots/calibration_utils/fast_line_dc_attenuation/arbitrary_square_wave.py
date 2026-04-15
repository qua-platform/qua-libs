import numpy as np
from typing import Any, List

from quam_builder.architecture.quantum_dots.components.mixins import VoltageMacroMixin

from quam.components.pulses import Pulse
from quam.core import quam_dataclass
from qualibration_libs.core import tracked_updates
from qualibrate import QualibrationNode

__all__ = ["SquareWave", "validate_and_add_square_wave"]


@quam_dataclass
class SquareWave(Pulse):
    """Gaussian pulse QUAM component.

    Args:
        amplitude (float): The amplitude of the pulse in volts.
        duration (int): The length of the pulse in samples.
        frequency_hz (int): The square wave frequency
        sampling_rate (int): The sampling rate in GS/s
    """

    frequency_hz: int = int(2e6)
    amplitude: float = 0.01
    sampling_rate: int = 1

    def waveform_function(self):
        period_ns = int(
            1 / self.frequency_hz * 1e9
        )  # For 1GSamples, this already gives us the number of points per period
        number_of_periods_in_duration = self.length // period_ns
        full_period_samples = (
            np.concatenate(
                [np.ones(period_ns // 2 * self.sampling_rate), -np.ones(period_ns // 2 * self.sampling_rate)]
            )
            * self.amplitude
        )
        return np.tile(full_period_samples, number_of_periods_in_duration)


def validate_and_add_square_wave(
    node: QualibrationNode,
    channel: VoltageMacroMixin,
    tracked_updates_list: List,
):
    if (
        not type(channel.physical_channel).__name__ == "VoltageGate"
    ):  # Not isinstance, to avoid import. Can change if necessary
        raise ValueError(
            f"Channel {channel.name}'s physical_channel is not a VoltageGate instance, but is {type(channel.physical_channel).__name__}."
        )
    elif channel.physical_channel.offset_parameter is None:
        raise ValueError(f"Channel {channel.name}'s physical_channel does not have an offset_parameter")

    with tracked_updates(channel.physical_channel.operations, auto_revert=False, dont_assign_to_none=True) as ops:
        pulse = SquareWave(
            length=max([s.readout_resonator.operations["readout"].length for s in node.namespace["sensors"]]),
            amplitude=node.parameters.square_wave_amplitude,
            frequency_hz=node.parameters.square_wave_frequency,
        )
        ops["square_wave"] = pulse
        tracked_updates_list.append(ops)
