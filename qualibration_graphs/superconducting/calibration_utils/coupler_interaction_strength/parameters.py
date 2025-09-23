from typing import Literal, Optional, List

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    qubit_pairs: Optional[List[str]] = ["qA1-qA2"]
    num_averages: int = 500
    # coupler_q1_q2:
    coupler_flux_min : float = -0.1
    coupler_flux_max : float = 0.1
    coupler_flux_step : float = 0.005
    idle_time_min : int = 16
    idle_time_max : int = 5000
    idle_time_step : int = 4
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    use_saved_detuning: bool = False
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

