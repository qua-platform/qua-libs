"""Parameters for the GEF duration sweep (all times in ns)."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for this duration sweep."""

    num_shots: int = Field(default=2000, ge=10)
    """Number of single-shot acquisitions per prepared state and duration."""

    min_duration_in_ns: int = Field(default=200, ge=16, multiple_of=4)
    """First readout duration in the inclusive sweep, in ns."""

    max_duration_in_ns: int = Field(default=2000, ge=16, multiple_of=4)
    """Last readout duration in the inclusive sweep, in ns."""

    duration_step_in_ns: int = Field(default=200, ge=4, multiple_of=4)
    """Spacing between readout durations, in ns."""

    outliers_threshold: float = Field(default=0.98, ge=0, le=1)
    """Optional minimum per-state Gaussian inlier fraction."""

    enforce_outliers_threshold: bool = False
    """If true, require g/e Gaussian inliers; otherwise retain them as diagnostics."""

    minimum_ge_fidelity: float = Field(default=0.90, ge=0, le=1)
    """Absolute minimum average correct assignment for prepared g and e."""

    ge_fidelity_tolerance: float = Field(default=0.01, ge=0, le=1)
    """Allowed drop from the best valid GE fidelity when defining its plateau."""

    @model_validator(mode="after")
    def validate_duration_range(self) -> "NodeSpecificParameters":
        """Require an ordered, evenly divisible duration sweep."""
        if self.max_duration_in_ns < self.min_duration_in_ns:
            raise ValueError("max_duration_in_ns must be >= min_duration_in_ns")
        if (self.max_duration_in_ns - self.min_duration_in_ns) % self.duration_step_in_ns:
            raise ValueError("The duration range must be divisible by duration_step_in_ns")
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for the three-state GEF duration optimization."""

    operation: Literal["readout_GEF"] = "readout_GEF"
    reset_type: Literal["thermal"] = "thermal"
    """Use thermal reset so the sweep does not depend on old discrimination thresholds."""


def duration_values(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the inclusive, validated readout-duration sweep."""
    # Validate again because local custom_param actions may assign fields separately.
    type(parameters).model_validate(parameters.model_dump())
    return np.arange(
        parameters.min_duration_in_ns,
        parameters.max_duration_in_ns + 1,
        parameters.duration_step_in_ns,
        dtype=int,
    )
