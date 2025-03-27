from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a T1 relaxation time experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
        min_wait_time_in_ns (int): Minimum wait time in nanoseconds. Must be a multiple of 4ns and larger than 16ns. Default is 16.
        max_wait_time_in_ns (int): Maximum wait time in nanoseconds. Must be a multiple of 4ns and larger than 16ns. Default is 100000.
        wait_time_num_points (int): Number of points fpr the wait time scan. Default is 51.
        log_or_linear_sweep (Literal["log", "linear"]): Type of sweep, either logarithmic or linear. Default is "linear".
    """

    num_averages: int = 1000
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 200000
    wait_time_num_points: int = 51
    log_or_linear_sweep: Literal["log", "linear"] = "log"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
