from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import Optional, Literal


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    operation: str = "x180_Gaussian"
    operation_amplitude_factor: Optional[float] = 1
    duration_in_ns: Optional[int] = 500
    frequency_span_in_mhz: float = 400
    frequency_step_in_mhz: float = 0.5
    flux_amp : float = 0.05
    update_lo: bool = False


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
