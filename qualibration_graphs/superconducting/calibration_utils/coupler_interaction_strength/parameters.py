from typing import List, Literal, Optional, ClassVar

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitPairExperimentNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    coupler_flux_min: float = -0.1
    coupler_flux_max: float = 0.1
    coupler_flux_step: float = 0.005
    idle_time_min: int = 16
    idle_time_max: int = 5000
    idle_time_step: int = 4
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    use_saved_detuning: bool = True


class Parameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitPairExperimentNodeParameters):

    targets_name: ClassVar[str] = "qubit_pairs"
