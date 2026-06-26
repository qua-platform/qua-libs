from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 13_xy8."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum half inter-pulse spacing in nanoseconds. Must be >= 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum half inter-pulse spacing in nanoseconds. Default is 10000 ns (10 µs)."""
    tau_step: int = 4
    """Step size for the half inter-pulse spacing sweep in nanoseconds. Default is 4 ns (1 clock cycle)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 13_xy8."""
