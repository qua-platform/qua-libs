import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
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
