from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    amp_range: float = 0.030
    """Range of amplitude variation around the nominal value, will scan between center - range and center + range. Default is 0.030."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    time_min_ns: float = 16
    time_max_ns: float = 1000
    time_step_ns: float = 16
    artificial_detuning_mhz: int = 1
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"
