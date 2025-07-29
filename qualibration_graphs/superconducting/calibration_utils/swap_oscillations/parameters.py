from typing import Optional, Literal, List
from qualibrate import NodeParameters


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ['q2-q4']
    num_averages: int = 100
    max_time_in_ns: int = 160
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    amp_range : float = 0.05
    amp_step : float = 0.001
    load_data_id: Optional[int] = None