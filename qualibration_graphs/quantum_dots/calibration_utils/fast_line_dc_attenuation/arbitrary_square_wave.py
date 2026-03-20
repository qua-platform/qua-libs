import numpy as np

from quam.components.pulses import Pulse
from quam.core import quam_dataclass


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
