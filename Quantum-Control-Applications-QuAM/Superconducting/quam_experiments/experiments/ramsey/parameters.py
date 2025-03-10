from typing import Literal, Optional, List
from qualibrate import NodeParameters


class RamseyParameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 3000
    wait_time_num_points: int = 500
    log_or_linear_sweep: Literal["log", "linear"] = "log"
    use_state_discrimination: bool = False
    multiplexed: bool = False
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    flux_point_joint_or_independent: str = "independent"  # e.g. "independent" or "joint"


def get_idle_times_in_clock_cycles(params: RamseyParameters):
    """
    Compute idle times (in clock cycles) using logarithmic or linear spacing.
    One clock cycle is 4 ns.
    """
    import numpy as np

    if params.log_or_linear_sweep == "log":
        idle_times = np.logspace(
            np.log10(params.min_wait_time_in_ns), np.log10(params.max_wait_time_in_ns), params.wait_time_num_points
        )
    else:
        idle_times = np.linspace(params.min_wait_time_in_ns, params.max_wait_time_in_ns, params.wait_time_num_points)
    # Convert idle times from ns to clock cycles (1 cycle = 4 ns)
    return (idle_times / 4).astype(int)


