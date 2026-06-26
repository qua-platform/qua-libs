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
    """Node-specific parameters for 18c_geometric_cz_phase_error_amplification."""

    num_shots: int = 100
    """Number of averages per point. Default is 100."""
    exchange_amplitude_center: Optional[float] = None
    """Fixed CZ exchange amplitude (barrier gate voltage, V).
    If None, the saved CZ voltage point for the first qubit pair is used."""
    max_num_cphase_gates: int = 20
    """Maximum CPhase repetition count (positive even integer).
    The sweep runs 2, 4, 6, …, max_num_cphase_gates."""
    num_phases: int = 21
    """Number of analysis phase points uniformly distributed over [0, 2π)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 18c_geometric_cz_phase_error_amplification."""