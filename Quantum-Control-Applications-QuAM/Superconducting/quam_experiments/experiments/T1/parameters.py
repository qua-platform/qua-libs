from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 1000
    """Number of averages to perform. Default is 1000."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Must be a multiple of 4ns and larger than 16ns. Default is 16."""
    max_wait_time_in_ns: int = 200000
    """Maximum wait time in nanoseconds. Must be a multiple of 4ns and larger than 16ns. Default is 200000."""
    wait_time_num_points: int = 51
    """Number of points for the wait time scan. Default is 51."""
    log_or_linear_sweep: Literal["log", "linear"] = "log"
    """Type of sweep, either "log" (logarithmic) or "linear". Default is "log"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
