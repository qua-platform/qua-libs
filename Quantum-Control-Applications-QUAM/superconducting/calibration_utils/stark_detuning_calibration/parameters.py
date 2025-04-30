from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 20
    """Number of averages. Default is 20."""
    operation: str = "x180"
    """Type of operation to perform. Default is "x180"."""
    frequency_span_in_mhz: float = 20
    """Span of frequencies to sweep in MHz. Default is 20 MHz."""
    frequency_step_in_mhz: float = 0.02
    """Step size for frequency sweep in MHz. Default is 0.02 MHz."""
    max_number_pulses_per_sweep: int = 20
    """Maximum number of pulses per sweep. Default is 20."""
    DRAG_setpoint: Optional[float] = -1.0
    """DRAG setpoint. Default is -1.0."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
