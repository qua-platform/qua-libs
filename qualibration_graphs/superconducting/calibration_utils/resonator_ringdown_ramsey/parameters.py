"""Parameters for resonator ring-down measured with a Ramsey probe."""

import numpy as np
from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Readout pulse, Ramsey phase sweep, and ring-down timing settings."""

    num_shots: int = Field(default=1000, ge=10)
    """Number of averages per delay and Ramsey analysis phase."""

    operation: str = "readout"
    """Calibrated resonator operation used to populate the resonator."""

    num_frame_rotations: int = Field(default=21, ge=5)
    """Number of Ramsey analysis phases covering one full turn."""

    use_state_discrimination: bool = True
    """Use calibrated state discrimination for the Ramsey readout."""

    min_ringdown_delay_ns: int = Field(default=16, ge=16)
    """First delay after the readout pulse, in ns."""

    max_ringdown_delay_ns: int = Field(default=3016, gt=16)
    """Last requested delay after the readout pulse, in ns."""

    ringdown_delay_step_ns: int = Field(default=80, ge=4)
    """Spacing between ring-down delays, in ns."""

    ramsey_idle_ns: int = Field(default=200, ge=16)
    """Fixed idle time between the two Ramsey x90 pulses, in ns."""


class Parameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitsExperimentNodeParameters):
    """Parameter set for 23d_resonator_ringdown_ramsey."""


def delay_values(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the inclusive delay sweep after checking the 4 ns timing grid."""
    values = (
        parameters.min_ringdown_delay_ns,
        parameters.max_ringdown_delay_ns,
        parameters.ringdown_delay_step_ns,
        parameters.ramsey_idle_ns,
    )
    if any(value % 4 for value in values):
        raise ValueError(f"All timing parameters must be divisible by 4 ns: {values}")
    return np.arange(
        parameters.min_ringdown_delay_ns,
        parameters.max_ringdown_delay_ns + parameters.ringdown_delay_step_ns // 2,
        parameters.ringdown_delay_step_ns,
        dtype=int,
    )


def frame_rotations(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return Ramsey analysis phases in turns, without the duplicate endpoint."""
    return np.arange(parameters.num_frame_rotations) / parameters.num_frame_rotations
