from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)

from quam_config import MacroParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for Ramsey 11a."""

    num_shots: int = 300
    """Number of averages to perform. Default is 100."""
    detuning_span_in_mhz: float = 5.0
    """Frequency detuning span. Default 5MHz."""
    detuning_step_in_mhz: float = 0.1
    """Frequency detuning step. Default 0.1MHz"""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    MacroParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 11c_ramsey_chevron (and related Ramsey chevron nodes)."""
