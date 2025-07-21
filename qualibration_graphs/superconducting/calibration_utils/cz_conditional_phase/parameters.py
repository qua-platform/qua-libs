from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, TwoQubitExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 50."""
    amp_range: float = 0.030
    """Range of amplitude variation around the nominal value. Default is 0.030."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    num_frames: int = 10
    """Number of frame rotation points for phase measurement. Default is 10."""
    plot_raw: bool = False
    """Whether to plot raw oscillation data. Default is False."""
    measure_leak: bool = False
    """Whether to measure leakage to the |f> state. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    TwoQubitExperimentNodeParameters,
):
    pass
