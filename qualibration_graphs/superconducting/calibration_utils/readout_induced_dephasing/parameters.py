"""Parameters for the directed readout-induced-dephasing measurement."""

import numpy as np
from pydantic import Field, model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Readout-amplitude and Ramsey-phase sweep settings."""

    num_shots: int = Field(default=1000, ge=10)
    """Number of averages per directed pair and sweep point."""

    operation: str = "readout"
    """Aggressor resonator operation played during the victim Ramsey interval."""

    min_amp_factor: float = Field(default=0.0, ge=0.0, lt=2.0)
    """Lower endpoint of the readout-amplitude-prefactor sweep."""

    max_amp_factor: float = Field(default=1.5, gt=0.0, lt=2.0)
    """Upper endpoint of the readout-amplitude-prefactor sweep."""

    amp_factor_step: float = Field(default=0.1, gt=0.0)
    """Spacing between readout-amplitude prefactors."""

    num_frame_rotations: int = Field(default=21, ge=5)
    """Number of Ramsey analysis phases covering one full turn."""

    report_amp_factor: float = Field(default=1.0, ge=0.0, lt=2.0)
    """Readout-amplitude factor at which dephasing and rotation are reported."""

    use_state_discrimination: bool = True
    """Acquire victim state probabilities; otherwise acquire demodulated IQ."""

    @model_validator(mode="after")
    def validate_sweeps(self) -> "NodeSpecificParameters":
        """Require an ordered amplitude sweep containing the reporting point."""
        if self.max_amp_factor < self.min_amp_factor:
            raise ValueError("max_amp_factor must be >= min_amp_factor")
        if self.max_amp_factor - self.min_amp_factor < self.amp_factor_step:
            raise ValueError("The amplitude sweep must contain at least two points")
        if not self.min_amp_factor <= self.report_amp_factor <= self.max_amp_factor:
            raise ValueError("report_amp_factor must lie inside the amplitude sweep")
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 23c_readout_induced_dephasing."""


def amplitude_factors(parameters: Parameters) -> np.ndarray:
    """Return an inclusive, QUA-safe readout-amplitude sweep."""
    type(parameters).model_validate(parameters.model_dump())
    values = np.arange(
        parameters.min_amp_factor,
        parameters.max_amp_factor + parameters.amp_factor_step / 2,
        parameters.amp_factor_step,
    )
    return values[values < 2.0]


def frame_rotations(parameters: Parameters) -> np.ndarray:
    """Analysis phases in turns, excluding the duplicate endpoint at one turn."""
    return np.arange(parameters.num_frame_rotations) / parameters.num_frame_rotations
