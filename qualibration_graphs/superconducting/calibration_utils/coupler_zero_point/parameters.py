from typing import Literal, Optional, List

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    qubit_pairs: Optional[List[str]] = ["qA1-qA2"]
    num_averages: int = 500
    coupler_flux_min : float = -0.05 #relative to the coupler set point
    coupler_flux_max : float = 0.05 #relative to the coupler set point
    coupler_flux_step : float = 0.0004
    qubit_flux_span : float = 0.026 # relative to the known/calculated detuning between the qubits
    qubit_flux_step : float = 0.0002
    pulse_duration_ns: int = 232
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
