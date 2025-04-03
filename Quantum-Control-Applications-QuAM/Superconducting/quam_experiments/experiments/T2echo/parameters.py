from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 70000
    """Maximum wait time in nanoseconds. Default is 70000."""
    wait_time_num_points: int = 51
    """Number of points for the wait time scan. Default is 51."""
    log_or_linear_sweep: Literal["log", "linear"] = "linear"
    """Type of sweep, either "log" (logarithmic) or "linear". Default is "linear"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
