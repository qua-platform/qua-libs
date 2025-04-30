from typing import Literal
import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    operation: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    """Type of operation to perform. Default is "x180"."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""
    max_number_pulses_per_sweep: int = 1
    """Maximum number of Rabi pulses per sweep. Default is 1."""
    update_x90: bool = True
    """Flag to update the x90 pulse amplitude. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_number_of_pulses(node_parameter: Parameters):
    if node_parameter.max_number_pulses_per_sweep > 1:
        if node_parameter.operation == "x180":
            N_pulses = np.arange(1, node_parameter.max_number_pulses_per_sweep, 2).astype("int")
        elif node_parameter.operation in ["x90", "-x90", "y90", "-y90"]:
            N_pulses = np.arange(2, node_parameter.max_number_pulses_per_sweep, 4).astype("int")
        else:
            raise ValueError(f"Unrecognized operation {node_parameter.operation}.")
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_number_pulses_per_sweep,
            node_parameter.max_number_pulses_per_sweep,
        ).astype("int")[::2]
    return N_pulses
