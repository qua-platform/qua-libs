from typing import Optional

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for node Qubit Spectroscopy Parity Diff"""
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "x90"
    """Type of operation to perform. Default is "x90"."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for Qubit Spectroscopy Parity Diff."""
