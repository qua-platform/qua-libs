from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a resonator vs amplitude experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 15 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.1 MHz.
        max_power_dbm (int): Maximum power level in dBm. Default is -30 dBm.
        min_power_dbm (int): Minimum power level in dBm. Default is -50 dBm.
        num_power_points (int): Number of points of the readout power axis. Default is 100.
        max_amp (float): Maximum readout amplitude for the experiment. Default is 0.1.
        derivative_crossing_threshold_in_hz_per_dbm (int): Threshold for derivative crossing in Hz/dBm. Default is -50000 Hz/dBm.
        derivative_smoothing_window_num_points (int): SSize of the window in number of points corresponding to the rolling average (number of points). Default is 10.
        moving_average_filter_window_num_points (int): Size of the moving average filter window (number of points). Default is 1.
        buffer_from_crossing_threshold_in_dbm (int): Buffer from the crossing threshold in dBm - the optimal readout power will be set to be this number in Db below the threshold. Default is 1 dBm.
    """

    num_averages: int = 100
    frequency_span_in_mhz: float = 15
    frequency_step_in_mhz: float = 0.1
    max_power_dbm: int = -25
    min_power_dbm: int = -50
    num_power_points: int = 100
    max_amp: float = 0.1
    derivative_crossing_threshold_in_hz_per_dbm: int = int(-50e3)
    derivative_smoothing_window_num_points: int = 10
    moving_average_filter_window_num_points: int = 5
    buffer_from_crossing_threshold_in_dbm: int = 1


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
