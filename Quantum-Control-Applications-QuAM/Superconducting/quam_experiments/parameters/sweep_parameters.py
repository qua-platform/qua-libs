from typing import Union
import numpy as np
from quam_experiments.experiments.ramsey.parameters import (
    NodeSpecificParameters as RamseyParameters,
)
from quam_experiments.experiments.T1.parameters import (
    NodeSpecificParameters as T1Parameters,
)


def get_idle_times_in_clock_cycles(
    node_parameters: Union[RamseyParameters, T1Parameters],
) -> np.ndarray:
    """
    Get the idle-times sweep axis according to the sweep type given by ``node.parameters.log_or_linear_sweep``.

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


def _get_idle_times_linear_sweep_in_clock_cycles(node_parameters: RamseyParameters):
    return (
        np.linspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)


def _get_idle_times_log_sweep_in_clock_cycles(node_parameters: RamseyParameters):
    return np.unique(
        np.geomspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)
