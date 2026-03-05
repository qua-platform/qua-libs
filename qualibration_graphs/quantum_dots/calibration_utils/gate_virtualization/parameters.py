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
    ramp_duration: int = 100
    """The ramp duration to each pixel. Set to zero for a step."""
    hold_duration: int = 1000
    """The dwell time on each pixel, after the ramp."""
    post_trigger_wait_ns: int = 10000
    """A pause in the QUA programme to allow the QDAC to reach the correct level."""
    pre_measurement_delay: int = 0
    """Extra delay (ns) inserted after the hold duration and before measurement."""


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
    sensor_gate_span: float = 0.05
    """Total voltage span of the sensor gate (X axis) sweep in volts."""
    sensor_gate_points: int = 201
    """Number of points along the sensor gate (X axis) sweep."""
    device_gate_span: float = 0.1
    """Total voltage span of the device gate (Y axis) sweep in volts."""
    device_gate_points: int = 201
    """Number of points along the device gate (Y axis) sweep."""
    sensor_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    device_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""


class VirtualPlungerParameters(GateVirtualizationBaseParameters):
    """Parameters for virtual plunger gate calibration."""

    plunger_device_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of plunger gate -> list of device gates (plungers or barriers)
    to scan against it.  Only neighbouring pairs need to be specified.
    Example: {"virtual_dot_1": ["virtual_dot_2", "barrier_12"]}.
    If None, must be generated from the machine (not yet implemented)."""
    plunger_gate_span: float = 0.1
    """Total voltage span of the plunger gate (X axis) sweep in volts."""
    plunger_gate_points: int = 201
    """Number of points along the plunger gate (X axis) sweep."""
    device_gate_span: float = 0.1
    """Total voltage span of the device gate (Y axis) sweep in volts."""
    device_gate_points: int = 201
    """Number of points along the device gate (Y axis) sweep."""
    plunger_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    device_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""


class BarrierCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for barrier compensation scans."""

    barrier_compensation_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of barrier gate -> list of gates (typically plungers) to scan
    against it for compensation.  Only local cross-talk pairs need to be
    specified.
    Example: {"barrier_12": ["virtual_dot_1", "virtual_dot_2"]}.
    If None, must be generated from the machine (not yet implemented)."""
    barrier_gate_span: float = 0.1
    """Total voltage span of the barrier gate (X axis) sweep in volts."""
    barrier_gate_points: int = 201
    """Number of points along the barrier gate (X axis) sweep."""
    compensation_gate_span: float = 0.1
    """Total voltage span of the compensation gate (Y axis) sweep in volts."""
    compensation_gate_points: int = 201
    """Number of points along the compensation gate (Y axis) sweep."""
    barrier_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    compensation_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""


def get_voltage_arrays(
    *,
    x_center: float,
    y_center: float,
    x_span: float,
    y_span: float,
    x_points: int,
    y_points: int,
):
    """Build X/Y voltage arrays from explicit sweep definitions.

    Parameters
    ----------
    x_center, y_center : float
        Sweep centres in volts.
    x_span, y_span : float
        Sweep spans in volts.
    x_points, y_points : int
        Number of points along each axis.
    """
    x_volts = np.linspace(x_center - x_span / 2, x_center + x_span / 2, x_points)
    y_volts = np.linspace(y_center - y_span / 2, y_center + y_span / 2, y_points)
    return x_volts, y_volts
