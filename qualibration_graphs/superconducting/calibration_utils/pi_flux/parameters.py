
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import Optional, Literal


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    operation: str = "x180_Gaussian"
    operation_amplitude_factor: float = 1.0
    """Amplitude factor for the operation. Default is 1.0."""
    duration_in_ns: int = 500
    frequency_span_in_mhz: float = 400
    frequency_step_in_mhz: float = 0.5
    qubit_detuning_in_mhz: int = 300
    update_lo: bool = False
    fitting_base_fractions: list = [0.4, 0.15, 0.05]
    """Base fractions for cascade fitting. Default is [0.4, 0.15, 0.05]."""
    update_state: bool = False
    """Whether to update system state with fit results. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass