from typing import List, Literal, Optional

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
    pre_measurement_delay: int = 0
    """Extra delay (ns) inserted after the hold duration and before measurement."""
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
