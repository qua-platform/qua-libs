from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.run_video_mode.video_mode_specific_parameters import VideoModeCommonParameters

from typing import List, Literal, Dict, Union, Callable


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    scan_pattern: Literal["raster", "switch_raster"] = "switch_raster"
    """The scanning pattern."""
    sensor_names: List[str] = None
    """List of sensor dot names to measure in your measurement."""
    x_axis_name: str = None
    """The name of the swept element in the X axis."""
    y_axis_name: str = None
    """The name of the swept element in the Y axis."""
    x_points: int = 101
    """Number of measurement points in the X axis."""
    y_points: int = 101
    """Number of measurement points in the Y axis."""
    x_span: float = 0.05
    """The X axis span in volts"""
    y_span: float = 0.05
    """The Y axis span in volts"""
    per_line_compensation: bool = True
    """Whether to send a compensation pulse at the end of each scan line."""
    perform_edge_analysis: bool = False
    """Whether to perform edge analysis on the data."""
    ramp_duration: int = 100
    """The ramp duration to each pixel. Set to zero for a step."""
    hold_duration: int = 1000
    """Dwell time on each point in nanoseconds. If using the QDAC, this must be slow enough."""
    pre_measurement_delay: int = 0
    """A deliberate delay time after the hold_duration and before the resonator measurement."""
    use_validation: bool = True
    """Whether to use validation with simulated data."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    VideoModeCommonParameters,
    NodeSpecificParameters,
):
    pass


class OPXQDACParameters(
    NodeParameters,
    VideoModeCommonParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    x_from_qdac: bool = False
    "Check to perform 2D map using the QDAC instead of the OPX"
    y_from_qdac: bool = False
    "Check to perform 2D map using the QDAC instead of the OPX"
    post_trigger_wait_ns: int = 10000
    """A pause in the QUA programme to allow the QDAC to get to the correct level."""


class SimulationParameters(
    NodeParameters,
    VideoModeCommonParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass


import numpy as np


def get_voltage_arrays(node):
    """Extract the X and Y voltage arrays from a given node."""
    x_span, x_center, x_points = node.parameters.x_span, 0, node.parameters.x_points
    y_span, y_center, y_points = node.parameters.y_span, 0, node.parameters.y_points
    x_volts, y_volts = np.linspace(x_center - x_span / 2, x_center + x_span / 2, x_points), np.linspace(
        y_center - y_span / 2, y_center + y_span / 2, y_points
    )
    return x_volts, y_volts


from .scan_modes import ScanMode


def _find_physical_dc_lists(
    scan_mode: ScanMode,
    virtual_dc_set: "VirtualDCSet",
    axis_name: str,
    axis_values: List[float],
) -> Dict[str, Union[List, np.ndarray]]:
    """Use the VirtualDCSet to yield a dictionary of physical dc_lists to use for the Qdac"""

    _, y_idxs = scan_mode.get_idxs(x_points=1, y_points=len(axis_values))
    ordered_axis_values = axis_values[y_idxs]

    full_physical_dicts = {name: [] for name in virtual_dc_set.channels.keys()}

    for value in ordered_axis_values:
        virtual_dict = {axis_name: float(value)}
        physical_dict = virtual_dc_set.resolve_voltages(virtual_dict)

        for physical_gate in virtual_dc_set.channels.keys():
            full_physical_dicts[physical_gate].append(physical_dict[physical_gate])

    # Check if the physical list is constant or not
    physical_lists = {
        name: arr for name, arr in physical_lists.items() if arr.size > 1 and not np.allclose(arr, arr[0], atol=1e-8)
    }
    return physical_lists


def prepare_dc_lists(
    node,
    virtual_dc_set_id: str,
    axis_name: str,
    axis_values: List[float],
    scan_mode: ScanMode,
) -> None:
    """
    Prepares the DC list attributes for the QDAC channel. This function assumes the use of the
    Qdac2 driver from qcodes_contrib_drivers. This also assumes that the VoltageGate objects have
    their QdacSpec objects configured with the qdac_output_port and opx_trigger_out.
    """
    virtual_dc_set = node.machine.virtual_dc_sets[virtual_dc_set_id]
    physical_dc_lists = _find_physical_dc_lists(scan_mode, virtual_dc_set, axis_name, axis_values)

    for name, voltages in physical_dc_lists.items():
        dc_list = node.machine.qdac.channel(virtual_dc_set.channels[name].qdac_spec.qdac_output_port).dc_list(
            voltages=voltages,
            dwell_s=node.parameters.qdac_dwell_time_us / 1e6,
            stepped=True,
        )
        # We want all the dc_lists associated with the same axis to start on the same trigger
        dc_list.start_on_external(
            trigger=virtual_dc_set.channels[physical_dc_lists.keys()[0]].qdac_spec.qdac_trigger_in
        )
