from typing import Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 17_geometric_cz_error_amplification."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    exchange_amplitude_center: Optional[float] = None
    """Center of the exchange amplitude sweep. If None, use the saved CZ voltage point."""
    exchange_amplitude_span: float = 0.02
    """Half span of the exchange amplitude sweep around the center (V). Default is 0.02."""
    exchange_amplitude_step: float = 0.002
    """Step size for the exchange amplitude sweep (V). Default is 0.002."""
    max_num_cphase_gates: int = 32
    """Maximum raw CPhase repetition count (must be a positive multiple of 2).
    The sweep runs 2, 4, 6, …, max_num_cphase_gates."""
    quadrature_signal_center: float = 0.5
    """Signal value corresponding to the center of the Ramsey I/Q circle. Default is 0.5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 17_geometric_cz_error_amplification."""
