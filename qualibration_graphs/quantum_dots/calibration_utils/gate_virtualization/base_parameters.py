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
    x_center: Optional[float] = None
    """Centre of the X axis sweep in volts.  When ``None`` and the axis is
    driven by the QDAC (``x_from_qdac=True``), the current DAC voltage is
    read from the machine and used as the centre.  For OPX-only axes the
    default is 0 (relative sweep)."""
    y_center: Optional[float] = None
    """Centre of the Y axis sweep in volts.  Same auto-detection logic as
    ``x_center``."""
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


def get_voltage_arrays(node, *, x_center: Optional[float] = None, y_center: Optional[float] = None):
    """Build the X and Y voltage arrays from the node's parameters.

    Parameters
    ----------
    node : QualibrationNode
        Provides ``node.parameters`` with span / points / center fields.
    x_center, y_center : float, optional
        Explicit overrides for the sweep centres.  When *None* the value
        from ``node.parameters`` is used; if that is also *None* the
        default is ``0.0`` (relative sweep for OPX).
    """
    p = node.parameters
    xc = x_center if x_center is not None else (p.x_center if p.x_center is not None else 0.0)
    yc = y_center if y_center is not None else (p.y_center if p.y_center is not None else 0.0)
    x_volts = np.linspace(xc - p.x_span / 2, xc + p.x_span / 2, p.x_points)
    y_volts = np.linspace(yc - p.y_span / 2, yc + p.y_span / 2, p.y_points)
    return x_volts, y_volts
