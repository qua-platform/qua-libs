from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)

from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Parameters for Ramsey 11a."""

    num_shots: int = 300
    """Number of averages to perform. Default is 100."""


class RamseyParameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 11a_ramsey."""

    frequency_detuning_in_mhz: float = 0.25
    """Frequency detuning in MHz. Default is 1.0 MHz."""


class RamseyDetuningParameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 11b_ramsey_detuning."""

    detuning_span_in_mhz: float = 5.0
    """Frequency detuning span. Default 5MHz."""
    detuning_step_in_mhz: float = 0.1
    """Frequency detuning step. Default 0.1MHz"""
    idle_time_ns: int = 100
    """Short idle time in ns (gives wide fringes for coarse localisation)."""
    idle_time_long_ns: int = 400
    """Long idle time in ns (gives narrow fringes for precision + T2* via amplitude ratio)."""


class RamseyChevronParameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 11c_ramsey_chevron (and related Ramsey chevron nodes)."""

    detuning_span_in_mhz: float = 5.0
    """Frequency detuning span. Default 5MHz."""
    detuning_step_in_mhz: float = 0.1
    """Frequency detuning step. Default 0.1MHz"""
