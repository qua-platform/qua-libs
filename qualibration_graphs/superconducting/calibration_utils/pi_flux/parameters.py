from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import Optional, Literal
from typing import List, Literal, Optional


class NodeSpecificParameters(RunnableParameters):
    """Parameters for 12a_pi_vs_flux
    """
    qubits: Optional[List[str]] = None
    operation: str = "x180_Gaussian"
    operation_amplitude_factor: float = 1.0
    duration_in_ns: int = 5000
    time_axis: Literal["linear", "log"] = "linear"
    time_step_in_ns: int = 48
    time_step_num: int = 200
    frequency_span_in_mhz: float = 200
    frequency_step_in_mhz: float = 0.4
    flux_amp: float = 0.06
    update_lo: bool = True
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    update_state: bool = False
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    timeout: int = 100
    multiplexed: bool = False
    reset_type_active_or_thermal: Literal["active", "thermal"] = "active"
    thermal_reset_extra_time_in_us: int = 10_000
    min_wait_time_in_ns: int = 32
    use_state_discrimination: bool = True



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass