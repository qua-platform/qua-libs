from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """The number of averages to perform. Default is 100."""
    sensor_names: List[str] = ["virtual_sensor_1"]
    """List of sensor dot names to measure in during the scan."""
    opx_fast_axis_name: Optional[str] = "virtual_dot_1"
    """The name of the component to be swept using the OPX. This should match a name of a gate axis in the VirtualGateSet."""
    dac_slow_axis_name: Optional[str] = "virtual_dot_2"
    """The name of the component to be stepped using the DAC. Right now, this is to be mapped to a VirtualDCSet axis, but this can/should be edited to suit your setup."""

    opx_offset: Optional[float] = None
    """Whether to center the OPX sweep around a particular value. The magnitude should be < 2.5V."""
    dac_offset: Optional[float] = None
    """The centre of the DAC sweep. If None, it will use the currently outputted value."""

    opx_points: int = 121
    """The number of measurement points on the fast/OPX axis. Default is 121."""
    dac_points: int = 81
    """The number of measurement points on the slow/DAC axis. Default is 81."""
    opx_span: float = 0.05
    """The span of the OPX sweep, in V."""
    dac_span: float = 0.05
    """The span of the DAC sweep, in V."""
    opx_ramp_duration: int = 100
    """The ramp duration to each pixel. Set to zero for a step instead of a ramp. Default is 100ns."""
    opx_hold_duration: int = 100
    """The hold time on each pixel, after the ramp but before the readout pulse is sent."""

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
    use_validation: bool = True
    """Use validation with simulated data. Default is True."""
    spiral_use_precomputed_scan: bool = False
    """Use the legacy precomputed spiral lookup lists. Default is False. Set to True to force list-based spiral generation."""

    post_trigger_wait_ns: int = 10000
    """Pause after each QDAC trigger, allowing the DAC to settle before readout [ns]. Default is 10000."""
    qdac_dwell_time_us: float = 200
    """Dwell time programmed into each QDAC dc_list step [µs]. Default is 200."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    """05c: 2D charge-stability map with an OPX fast axis and external-DAC slow axis."""
