from typing import Optional, Literal

import numpy as np
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)

from calibration_utils.common_utils.experiment import ParityDiffAnalysisParameters, HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for node Qubit Spectroscopy Parity Diff"""
    num_shots: int = 300
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 4
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.01
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for Qubit Spectroscopy Parity Diff."""


class InitNodeSpecificParameters(RunnableParameters): 
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


class InitDurParameters(
    InitNodeSpecificParameters,
    NodeParameters, 
    CommonNodeParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
): 
    """Params for the init node"""
