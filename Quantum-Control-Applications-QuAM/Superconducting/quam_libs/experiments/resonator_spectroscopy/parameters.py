from typing import Optional

from quam_libs.experiments.node import SimulatableNodeParameters


class Parameters(SimulatableNodeParameters):
    num_averages: int = 100
    frequency_span_in_mhz: float = 30.0
    frequency_step_in_mhz: float = 0.1
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True    


