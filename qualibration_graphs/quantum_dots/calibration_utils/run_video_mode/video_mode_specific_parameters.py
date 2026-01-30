from qualibrate.parameters import RunnableParameters
from typing import List, Optional

class VideoModeCommonParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    virtual_gate_set_id: str = None
    """Name of the VirtualGateSet to measure using Video Mode."""
    x_axis_name: str = None
    """Name of physical or virtual X axis."""
    y_axis_name: str = None
    """Name of physical or virtual Y axis."""
    sensor_names: Optional[List[str]] = None
    """Sensors that you would like to sweep"""
    dc_control: bool = False
    """Whether to include DC Control channels in Video Mode"""
