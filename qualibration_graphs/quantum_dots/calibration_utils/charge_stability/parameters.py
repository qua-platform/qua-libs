from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    run_in_video_mode: bool = False
    """Whether to run this measurement in Video Mode."""
    virtual_gate_set_id: str = None
    """Name of the VirtualGateSet of this measurement."""
    sensor_names: List[str] = None
    """List of sensor dot names to measure in your measurement."""
    x_axis_name: str = None
    """The name of the swept element in the X axis."""
    y_axis_name: str = None
    """The name of the swept element in the Y axis."""
    x_points: int = 201
    """Number of measurement points in the X axis."""
    y_points: int = 201
    """Number of measurement points in the Y axis."""
    x_span: float = 0.05
    """The X axis span in volts"""
    y_span: float = 0.05
    """The Y axis span in volts"""
    dc_control: bool = False
    """Includes VoltageControlComponent in Video Mode."""
    use_validation: bool = True
    """Whether to use validation with simulated data."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

import numpy as np
def get_voltage_arrays(node): 
    """Extract the X and Y voltage arrays from a given node."""
    x_span, x_center, x_points = node.parameters.x_span, 0, node.parameters.x_points
    y_span, y_center, y_points = node.parameters.y_span, 0, node.parameters.y_points
    x_volts, y_volts = np.linspace(x_center - x_span/2, x_center + x_span/2, x_points), np.linspace(y_center - y_span/2, y_center + y_span/2, y_points)
    return x_volts, y_volts
