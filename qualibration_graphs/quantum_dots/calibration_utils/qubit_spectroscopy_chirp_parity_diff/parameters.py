from typing import Optional

import numpy as np
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)

from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Parameters for node Qubit Spectroscopy Chirp Parity Diff"""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 500
    """Span of frequencies to chirp through in MHz. Default is 500 MHz."""
    frequency_step_in_mhz: float = 1
    """Frequency chirp width in MHz. Default is 1 MHz."""
    operation: str = "gaussian"
    """Type of operation to perform. Must match a key in qubit.xy.operations. Default is "gaussian"."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = 1000
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""
    fit_peak: bool = False
    """Whether to attempt a peak fit on the data. Default False"""
    signal_threshold: float = 0.25
    """Necessary height of signal to locate qubit frequency."""
    use_simulated_data: bool = False
    """Whether to run the node and produce simulated data rather than measuring via the OPX. Default False."""
    parity_pre_measurement: bool = True
    """Whether to measure parity before the drive pulse (empty → measure). Default True for chirp spectroscopy."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for Qubit Spectroscopy Chirp Parity Diff."""
