from typing import Dict, List, Literal, Optional

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from calibration_utils.run_video_mode.video_mode_specific_parameters import (
    VideoModeCommonParameters,
)


class GateVirtualizationNodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform."""
    scan_pattern: Literal["raster", "switch_raster", "spiral"] = "switch_raster"
    """The scanning pattern."""
    per_line_compensation: bool = True
    """Whether to send a compensation pulse at the end of each scan line."""
    sensor_names: Optional[List[str]] = None
    """List of sensor dot names to measure."""
    x_axis_name: Optional[str] = None
    """The name of the swept element in the X axis."""
    y_axis_name: Optional[str] = None
    """The name of the swept element in the Y axis."""
    x_points: int = 201
    """Number of measurement points in the X axis."""
    y_points: int = 201
    """Number of measurement points in the Y axis."""
    x_span: float = 0.05
    """The X axis span in volts."""
    y_span: float = 0.05
    """The Y axis span in volts."""
    ramp_duration: int = 100
    """The ramp duration to each pixel. Set to zero for a step."""
    hold_duration: int = 1000
    """The dwell time on each pixel, after the ramp."""
    pre_measurement_delay: int = 0
    """A deliberate delay time after the hold_duration and before the resonator measurement."""
    x_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    y_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""
    post_trigger_wait_ns: int = 10000
    """A pause in the QUA programme to allow the QDAC to reach the correct level."""


class GateVirtualizationBaseParameters(
    NodeParameters,
    VideoModeCommonParameters,
    CommonNodeParameters,
    GateVirtualizationNodeSpecificParameters,
):
    """Base parameter class for all gate virtualization nodes."""

    pass


class SensorCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for sensor gate vs device gate compensation scans."""

    sensor_device_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of sensor gate -> list of device gates to scan against it.
    Only local cross-talk pairs need to be specified.
    Example: {"virtual_sensor_1": ["virtual_dot_1", "virtual_dot_2"]}.
    If None, must be generated from the machine (not yet implemented)."""


class VirtualPlungerParameters(GateVirtualizationBaseParameters):
    """Parameters for virtual plunger gate calibration."""

    plunger_device_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of plunger gate -> list of device gates (plungers or barriers)
    to scan against it.  Only neighbouring pairs need to be specified.
    Example: {"virtual_dot_1": ["virtual_dot_2", "barrier_12"]}.
    If None, must be generated from the machine (not yet implemented)."""


class BarrierCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for barrier compensation scans."""

    barrier_compensation_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of barrier gate -> list of gates (typically plungers) to scan
    against it for compensation.  Only local cross-talk pairs need to be
    specified.
    Example: {"barrier_12": ["virtual_dot_1", "virtual_dot_2"]}.
    If None, must be generated from the machine (not yet implemented)."""


def get_voltage_arrays(node):
    """Extract the X and Y voltage arrays from a given node's parameters."""
    x_span, x_center, x_points = node.parameters.x_span, 0, node.parameters.x_points
    y_span, y_center, y_points = node.parameters.y_span, 0, node.parameters.y_points
    x_volts = np.linspace(x_center - x_span / 2, x_center + x_span / 2, x_points)
    y_volts = np.linspace(y_center - y_span / 2, y_center + y_span / 2, y_points)
    return x_volts, y_volts
