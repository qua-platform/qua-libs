import numpy as np
from typing import Dict, List, Union, Sequence

from quam.components.pulses import SquarePulse
from quam_builder.architecture.quantum_dots.components import VirtualDCSet
from qualibrate.core import QualibrationNode

from calibration_utils.charge_stability_opx import get_axis_names_and_validate, ScanMode
from .helper_utils import get_voltage_arrays

__all__ = ["prepare_dc_lists", "select_scan_trigger"]


TRIGGER_PULSE_LENGTH_NS = 1000


def _find_physical_dc_lists(
    virtual_dc_set: VirtualDCSet,
    axis_name: str,
    axis_values: List[float],
) -> Dict[str, Union[List, np.ndarray]]:
    """Use the VirtualDCSet to yield a dictionary of physical dc_lists to use for the Qdac"""

    full_physical_dicts = {name: [] for name in virtual_dc_set.channels.keys()}

    for value in axis_values:
        virtual_dict = {axis_name: float(value)}
        physical_dict = virtual_dc_set.resolve_voltages(virtual_dict)

        for physical_gate in virtual_dc_set.channels.keys():
            full_physical_dicts[physical_gate].append(physical_dict[physical_gate])

    # Check if the physical list is constant or not
    physical_lists = {
        name: arr
        for name, arr in full_physical_dicts.items()
        if len(arr) > 1 and not np.allclose(arr, arr[0], atol=1e-8)
    }
    return physical_lists


def _find_physical_dc_list_per_pixel(
    virtual_dc_set: VirtualDCSet,
    x_axis_name: str,
    y_axis_name: str,
    x_axis_values: Sequence[float],
    y_axis_values: Sequence[float],
    scan_mode: ScanMode,
):
    """
    Find a dictionary of physical dc lists to output from the QDAC, in the event that the desired outputs are both
    from the QDAC (not mixed).

    This reproduces the scan pattern and resolves each pixel, appending the value to a dictionary of physical outputs required.

    In order to not saturate the memory of the QDAC, we also check (to a tolerance) whether the produced physical DC list of all
    channel are actually varying or not.
    """
    x_idxs, y_idxs = scan_mode.get_idxs(len(x_axis_values), len(y_axis_values))
    x_path = np.asarray(x_axis_values)[x_idxs]
    y_path = np.asarray(y_axis_values)[y_idxs]

    physical_dc_lists = {ch: [] for ch in list(virtual_dc_set.channels.keys())}

    for x, y in zip(x_path, y_path):
        voltages_dict = {x_axis_name: x, y_axis_name: y}

        resolved_pixel_voltage = virtual_dc_set.resolve_voltages(voltages_dict)

        for name, value in resolved_pixel_voltage.items():
            physical_dc_lists[name].append(value)

    # Check if the physical list is constant or not
    physical_lists = {
        name: arr for name, arr in physical_dc_lists.items() if len(arr) > 1 and not np.allclose(arr, arr[0], atol=1e-8)
    }
    return physical_lists


def _get_offset(
    axis_name: str,
    offset: None | float,
    virtual_dc_set: VirtualDCSet,
):
    if offset is None:  # This means that the user would like to centre their DAC sweep around the current value
        current_dac_value = virtual_dc_set.get_voltage(axis_name, requery=True)
        return current_dac_value
    else:  # User specified x_offset
        return offset


def _find_trigger_in(
    axis_name: str,
    physical_dc_lists: Dict[str, List[float]],
    virtual_dc_set: VirtualDCSet,
) -> int:
    trigger = None
    for ch_name in physical_dc_lists.keys():
        spec = getattr(virtual_dc_set.channels[ch_name], "qdac_spec", None)
        if spec is not None:
            trig = getattr(spec, "qdac_trigger_in", None)
            if trig is not None:
                trigger = trig
                break
    if trigger is None:
        raise ValueError(f"No trigger found for the physical outputs associated with the axis {axis_name}")
    return trigger


def _resolve_trigger_gate(
    axis_name: str,
    physical_dc_lists: Dict[str, List[float]],
    virtual_dc_set: VirtualDCSet,
) -> tuple[int, str]:
    for ch_name in physical_dc_lists.keys():
        gate = virtual_dc_set.channels[ch_name]
        spec = getattr(gate, "qdac_spec", None)
        if spec is None:
            continue
        trig = getattr(spec, "qdac_trigger_in", None)
        if trig is None:
            continue
        if len(getattr(gate, "digital_outputs", {}) or {}) == 0:
            raise ValueError(
                f"Physical gate '{ch_name}' provides qdac_trigger_in={trig}, "
                "but it has no digital_outputs to emit an OPX trigger pulse."
            )
        if "trigger" not in gate.operations:
            gate.operations["trigger"] = SquarePulse(
                amplitude=0.0,
                length=TRIGGER_PULSE_LENGTH_NS,
                digital_marker="ON",
            )
        return trig, ch_name
    raise ValueError(f"No trigger-capable physical gate found for axis '{axis_name}'.")


def _load_physical_dc_lists(
    node: QualibrationNode,
    physical_dc_lists: Dict[str, List[float]],
    virtual_dc_set: VirtualDCSet,
    trig: int,
):
    dacs = getattr(node.machine, "dacs", None)
    if dacs:
        for name, voltages in physical_dc_lists.items():
            voltage_gate_channel = virtual_dc_set.channels[name]
            dac_name: str = voltage_gate_channel.dac_spec.dac_name
            output_port: int = voltage_gate_channel.dac_spec.output_port

            dac_info = dacs.get(dac_name, None)
            if dac_info is None:
                raise ValueError(f"Dac {dac_name} not found. Please double check.")

            qdac_channel = getattr(dac_info["driver"], dac_info["channel_method"])(output_port)

            dc_list = qdac_channel.dc_list(
                voltages=voltages,
                dwell_s=node.parameters.qdac_dwell_time_us / 1e6,
                stepped=True,
            )
            dc_list.start_on_external(trigger=trig)


def prepare_dc_lists(
    node: QualibrationNode,
):
    x_ext = node.parameters.x_from_qdac
    y_ext = node.parameters.y_from_qdac

    x_axis_name, y_axis_name, vgs_id = get_axis_names_and_validate(node)
    virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]

    x_volts_no_offset, y_volts_no_offset = get_voltage_arrays(node)
    x_offset = node.parameters.x_offset
    y_offset = node.parameters.y_offset

    # If x_ext, only prepare the X dc_list
    if x_ext and not y_ext:
        resolved_x_offset = _get_offset(x_axis_name, x_offset, virtual_dc_set)
        node.namespace["resolved_offsets"] = node.namespace.get("resolved_offsets", {})
        node.namespace["resolved_offsets"][x_axis_name] = resolved_x_offset
        # x_array = x_volts_no_offset + resolved_x_offset

        # Find the right x_array based on the scan mode (Raster vs SwitchRaster)
        scan_mode = node.namespace["scan_mode"]
        if scan_mode.name == "spiral": 
            raise ValueError("Spiral scan pattern does not work with mixed axes. Please choose another.")
        x_array = scan_mode.get_y_axis_order(x_volts_no_offset + resolved_x_offset)

        physical_dc_lists = _find_physical_dc_lists(virtual_dc_set, x_axis_name, x_array)
        trig, trigger_gate_name = _resolve_trigger_gate(x_axis_name, physical_dc_lists, virtual_dc_set)
        node.namespace["trigger_gate_name"] = trigger_gate_name
        _load_physical_dc_lists(node, physical_dc_lists, virtual_dc_set, trig)
        return

    if y_ext and not x_ext:
        resolved_y_offset = _get_offset(y_axis_name, y_offset, virtual_dc_set)
        node.namespace["resolved_offsets"] = node.namespace.get("resolved_offsets", {})
        node.namespace["resolved_offsets"][y_axis_name] = resolved_y_offset
        # y_array = y_volts_no_offset + resolved_y_offset
        
        # Find the right x_array based on the scan mode (Raster vs SwitchRaster)
        scan_mode = node.namespace["scan_mode"]
        if scan_mode.name == "spiral": 
            raise ValueError("Spiral scan pattern does not work with mixed axes. Please choose another.")
        y_array = scan_mode.get_y_axis_order(y_volts_no_offset + resolved_y_offset)

        physical_dc_lists = _find_physical_dc_lists(virtual_dc_set, y_axis_name, y_array)
        trig, trigger_gate_name = _resolve_trigger_gate(y_axis_name, physical_dc_lists, virtual_dc_set)
        node.namespace["trigger_gate_name"] = trigger_gate_name
        _load_physical_dc_lists(node, physical_dc_lists, virtual_dc_set, trig)
        return

    if x_ext and y_ext:
        resolved_x_offset = _get_offset(x_axis_name, x_offset, virtual_dc_set)
        resolved_y_offset = _get_offset(y_axis_name, y_offset, virtual_dc_set)

        node.namespace["resolved_offsets"] = node.namespace.get("resolved_offsets", {})
        node.namespace["resolved_offsets"][x_axis_name] = resolved_x_offset
        node.namespace["resolved_offsets"][y_axis_name] = resolved_y_offset

        x_array = x_volts_no_offset + resolved_x_offset
        y_array = y_volts_no_offset + resolved_y_offset

        physical_dc_lists = _find_physical_dc_list_per_pixel(
            virtual_dc_set, x_axis_name, y_axis_name, x_array, y_array, node.namespace["scan_mode"]
        )
        trig, trigger_gate_name = _resolve_trigger_gate(x_axis_name, physical_dc_lists, virtual_dc_set)
        node.namespace["trigger_gate_name"] = trigger_gate_name
        _load_physical_dc_lists(node, physical_dc_lists, virtual_dc_set, trig)


def select_scan_trigger(
    x_axis_name: str,
    y_axis_name: str,
    virtual_dc_set: VirtualDCSet,
) -> tuple[int, str]:
    """Probe X then Y with dummy values to find which physical channels each
    touches, and return the trigger plus the physical gate name that can emit it."""
    for axis_name in (x_axis_name, y_axis_name):
        probe_lists = _find_physical_dc_lists(virtual_dc_set, axis_name, [0.0, 1.0])
        try:
            return _resolve_trigger_gate(axis_name, probe_lists, virtual_dc_set)
        except ValueError:
            continue
    raise ValueError(f"No trigger found for physical channels touched by '{x_axis_name}' or '{y_axis_name}'.")
