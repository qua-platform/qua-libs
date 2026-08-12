from typing import Optional, Literal

import numpy as np
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters
from calibration_utils.measurement_utils import ParityDiffAnalysisParameters


class NodeSpecificParameters(RunnableParameters): 
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    ramp_duration_start: int = 100
    """Duration (hold or ramp) start."""
    ramp_duration_stop: int = 10000
    """Duration (hold or ramp) start."""
    ramp_duration_step: int = 100
    """Duration (hold or ramp) start."""
    hold_duration_start: int = 100
    """Duration (hold or ramp) start."""
    hold_duration_stop : int= 10000
    """Duration (hold or ramp) start."""
    hold_duration_step: int = 100
    """Duration (hold or ramp) start."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""


class Parameters(
    NodeSpecificParameters,
    NodeParameters, 
    CommonNodeParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
): 
    """Params for the init node"""
