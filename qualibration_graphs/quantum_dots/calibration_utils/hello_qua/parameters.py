from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    num_points: int = 51
    """Resolution of 1d sweep"""
    quantum_dots: List[str] = None
    """Quantum Dots to sweep in your measurement."""
    v_center: float = 0
    """Where to centre gate sweep."""
    v_span: float = 0.01
    """Gate Sweep Span in V."""
    sensor_names: List[str] = None
    """List of sensor dot names to measure in your measurement."""
    virtual_gate_set_id: str = None
    """Name of the VirtualGateSet of this measurement."""
    run_in_video_mode: bool = False
    """Whether to run this measurement in Video Mode."""
    dc_control: bool = False
    """Includes VoltageControlComponent in Video Mode."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
