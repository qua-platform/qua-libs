from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "saturation"
    """Type of operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""
    target_peak_width: float = 3e6
    """Target peak width in Hz. Default is 3e6 Hz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
