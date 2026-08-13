from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.run_video_mode.video_mode_utils import VideoModeCommonParameters

from typing import List, Literal, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """The number of averages to perform. Default is 100."""
    scan_pattern: Literal["raster", "switch_raster", "spiral"] = "switch_raster"
    """The scanning pattern. Default is switch_raster."""
    sensor_names: List[str] = ["virtual_sensor_1"]
    """List of sensor dot names to measure in during the scan."""
    x_axis_name: Optional[str] = "virtual_dot_1"
    """The name of the element swept on the X axis."""
    y_axis_name: Optional[str] = "virtual_dot_2"
    """The name of the element swept on the Y axis."""
    x_points: int = 121
    """The number of measurement points on the X axis. Default is 121."""
    y_points: int = 121
    """The number of measurement points on the Y axis. Default is 121."""
    x_span: float = 0.05
    """The X axis span in volts. Default is 1V."""
    y_span: float = 0.05
    """The Y axis span in volts. Default is 1V."""
    ramp_duration: int = 100
    """The ramp duration to each pixel. Set to zero for a step instead of a ramp. Default is 100ns."""
    hold_duration: int = 100
    """The hold time on each pixel, after the ramp but before the readout pulse is sent."""
    x_offset: Optional[float] = None
    """The center of the X axis sweep. If dc_control = True, then this will be applied to the external source. Else, it will be applied by the OPX."""
    y_offset: Optional[float] = None
    """The center of the Y axis sweep. If dc_control = True, then this will be applied to the external source. Else, it will be applied by the OPX."""
    dc_control: bool = False
    """Apply sweep center via external DC (VirtualDCSet) instead of OPX offset."""
    per_line_compensation: bool = True
    """Send a compensation pulse at the end of each scan line. Default is True."""
    max_compensation_voltage: float = 0.05
    """The maximum voltage for the compensation pulse. Default is 0.05V."""
    plot_points: bool = False
    """Plot the existing points saved in the VirtualGateSet. Default is False."""
    perform_edge_analysis: bool = False
    """Perform edge analysis on the data. Default is False."""
    per_line_wait: int = 0
    """The wait time at the start of each line, in order to allow the electrostatics to settle. Default is 0ns."""
    spiral_use_precomputed_scan: bool = False
    """Use the legacy precomputed spiral lookup lists. Default is False. Set to True to force list-based spiral generation."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    VideoModeCommonParameters,
    NodeSpecificParameters,
):
    """05a: OPX charge stability map with optional video mode."""


class OPXuDACParameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    """05e: OPX charge stability map using user-specific DAC voltage sequences."""
    plot_pca: bool = True
