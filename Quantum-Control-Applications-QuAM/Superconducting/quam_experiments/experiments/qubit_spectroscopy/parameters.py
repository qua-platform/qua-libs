from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    FluxControlledNodeParameters,
    CommonNodeParameters,
)

class QubitSpectroscopyParameters(RunnableParameters):
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.5
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 100
    frequency_step_in_mhz: float = 0.25
    target_peak_width: Optional[float] = 3e6

class Parameters(
    NodeParameters,
    FluxControlledNodeParameters,
    CommonNodeParameters,
    QubitSpectroscopyParameters,
    QubitsExperimentNodeParameters,
):
    pass
