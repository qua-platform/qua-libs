"""Parameters shared by the GE and GEF duration sweeps (all times in ns)."""

from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = Field(default=2000, ge=10)
    min_duration_in_ns: int = Field(default=200, ge=16, multiple_of=4)
    max_duration_in_ns: int = Field(default=2000, ge=16, multiple_of=4)
    duration_step_in_ns: int = Field(default=200, ge=4, multiple_of=4)
    outliers_threshold: float = Field(default=0.98, ge=0, le=1)
    """Minimum fraction within log(0.01) of the peak GMM density, as in 08b."""

    @model_validator(mode="after")
    def validate_duration_range(self):
        if self.max_duration_in_ns < self.min_duration_in_ns:
            raise ValueError("max_duration_in_ns must be >= min_duration_in_ns")
        if (self.max_duration_in_ns - self.min_duration_in_ns) % self.duration_step_in_ns:
            raise ValueError("The duration range must be divisible by duration_step_in_ns")
        return self


class Parameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitsExperimentNodeParameters):
    operation: Literal["readout"] = "readout"
    reset_type: Literal["thermal"] = "thermal"
    """Use thermal reset so the sweep does not depend on old discrimination thresholds."""


class GEFParameters(Parameters):
    operation: Literal["readout_GEF"] = "readout_GEF"


def duration_values(parameters):
    # Validate again because local custom_param actions may assign fields separately.
    type(parameters).model_validate(parameters.model_dump())
    return np.arange(
        parameters.min_duration_in_ns,
        parameters.max_duration_in_ns + 1,
        parameters.duration_step_in_ns,
        dtype=int,
    )
