from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for Ramsey 10a."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    gap_wait_time_in_ns: int = 128
    """Wait time between initialization and qubit pulse in nanoseconds. Default is 128 ns."""


class RamseyParameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 10a_ramsey_parity_diff."""

    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""


class RamseyDetuningParameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 10b_ramsey_detuning_parity_diff."""

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
):
    """Parameter set for 10b_ramsey_detuning_parity_diff."""

    detuning_span_in_mhz: float = 5.0
    """Frequency detuning span. Default 5MHz."""
    detuning_step_in_mhz: float = 0.1
    """Frequency detuning step. Default 0.1MHz"""
