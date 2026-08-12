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
    rotation_scale: int = 1
    """The number of pi rotations to perform. By default, this is a pi pulse. For higher integers, this will scale the amplitude as a tracked change."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for Qubit Spectroscopy Parity Diff."""
