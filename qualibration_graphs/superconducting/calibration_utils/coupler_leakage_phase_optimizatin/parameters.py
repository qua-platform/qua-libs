from typing import Literal, Optional, List

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    qubit_pairs: Optional[List[str]] = ["qA1-qA2"]
    num_averages: int = 10
    # coupler_q1_q2:
    coupler_flux_min : float = -0.01 # relative to the coupler set point
    coupler_flux_max : float = 0.03 # relative to the coupler set point

    coupler_flux_step : float = 0.001

    qubit_flux_min : float = -0.03 # relative to the qubit pair detuning
    qubit_flux_max : float = 0.03 # relative to the qubit pair detuning
    qubit_flux_step : float = 0.001

    use_state_discrimination: bool = True
    num_frames : int = 10#20
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    use_saved_detuning: bool = False
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
