from typing import Literal, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class BasePowerRabiParameters(RunnableParameters):
    """Parameters shared by both 04b (GE power Rabi) and 12b (EF power Rabi) nodes."""

    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""


class NodeSpecificParameters(BasePowerRabiParameters):
    """04b-specific parameters (GE power Rabi with optional error amplification)."""

    operation: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    """Type of operation to perform. Default is "x180"."""
    max_number_pulses_per_sweep: int = 1
    """Maximum number of Rabi pulses per sweep (error amplification). Default is 1."""
    update_x90: bool = True
    """Flag to update the x90 pulse amplitude after calibrating x180. Default is True."""


class EfNodeSpecificParameters(BasePowerRabiParameters):
    """12b EF-specific parameters (no operation choice / error amplification knobs)."""

    # Potential future EF-specific fields can be added here
    pass


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 04b_power_rabi."""


class EfParameters(
    NodeParameters,
    CommonNodeParameters,
    EfNodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 12b_power_rabi_ef (EF transition)."""


@runtime_checkable
class HasErrorAmplification(Protocol):
    """Structural typing for objects supporting error amplification controls."""

    max_number_pulses_per_sweep: int
    operation: str


def get_number_of_pulses(node_parameter: BasePowerRabiParameters):
    """Return array of number of pulses for error amplification.

    For EF node (12b) the default behaviour is a single pulse sweep (equivalent to max_number_pulses_per_sweep = 1).
    """
    # If the parameter object lacks error amplification attributes, default to single pulse.
    if not isinstance(node_parameter, HasErrorAmplification):
        return np.array([1], dtype=int)

    if node_parameter.max_number_pulses_per_sweep > 1:
        if node_parameter.operation == "x180":
            N_pulses = np.arange(1, node_parameter.max_number_pulses_per_sweep, 2).astype(int)
        elif node_parameter.operation in ["x90", "-x90", "y90", "-y90"]:
            N_pulses = np.arange(2, node_parameter.max_number_pulses_per_sweep, 4).astype(int)
        else:
            raise ValueError(f"Unrecognized operation {node_parameter.operation}.")
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_number_pulses_per_sweep,
            node_parameter.max_number_pulses_per_sweep,
        ).astype(int)[::2]
    return N_pulses
