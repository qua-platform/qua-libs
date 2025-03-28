from typing import Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 30000
    wait_time_num_points: int = 500
    log_or_linear_sweep: Literal["log", "linear"] = "log"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_idle_times_in_clock_cycles(
    node_parameters: NodeSpecificParameters,
) -> np.ndarray:
    """
    Get the idle-times sweep axis according to the sweep type.

    The dephasing time sweep is in units of clock cycles (4ns).
    The minimum is 4 clock cycles.
    """
    if node_parameters.log_or_linear_sweep == "linear":
        idle_times = _get_idle_times_linear_sweep_in_clock_cycles(node_parameters)
    elif node_parameters.log_or_linear_sweep == "log":
        idle_times = _get_idle_times_log_sweep_in_clock_cycles(node_parameters)
    else:
        raise ValueError(f"Expected sweep type to be 'log' or 'linear', got {node_parameters.log_or_linear_sweep}")

    return idle_times


def _get_idle_times_linear_sweep_in_clock_cycles(
    node_parameters: NodeSpecificParameters,
):
    return (
        np.linspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)


def _get_idle_times_log_sweep_in_clock_cycles(node_parameters: NodeSpecificParameters):
    return np.unique(
        np.geomspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)
