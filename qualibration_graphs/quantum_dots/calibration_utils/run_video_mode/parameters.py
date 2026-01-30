from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List, Literal

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    virtual_gate_set_id: str = None
    """Name of the VirtualGateSet to measure using Video Mode."""
    x_axis_name: str = None
    """Name of the X axis in your VirtualGateSet. Can be a physical gate id or virtual gate id."""
    y_axis_name: str = None
    """Name of the Y axis in your VirtualGateSet. Can be a physical gate id or virtual gate id."""
    x_axis_mode: Literal["Voltage", "Frequency", "Amplitude"] = "Voltage"
    """Axis type of the X axis. Can be 'Voltage', 'Frequency', 'Amplitude'."""
    y_axis_mode: Literal["Voltage", "Frequency", "Amplitude"] = "Voltage"
    """Axis type of the Y axis. Can be 'Voltage', 'Frequency', 'Amplitude'."""
    sensor_names: List[str] = None
    """List of sensor names to include in the measurement. """
    dc_control: bool = False
    """Whether to include DC Control channels in Video Mode"""
    result_type: Literal["I", "Q", "Amplitude", "Phase"] = "I"
    """Result type to display. Can be 'I', 'Q', 'Amplitude', 'Phase'."""
    

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
