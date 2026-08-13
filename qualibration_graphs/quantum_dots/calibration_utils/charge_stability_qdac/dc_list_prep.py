import numpy as np
from typing import Dict, List, Union

from quam_builder.architecture.quantum_dots.components import VirtualDCSet

__all__ = ["prepare_dc_lists"]


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


def prepare_dc_lists(
    node,
    virtual_dc_set_id: str,
    axis_name: str,
    axis_values: List[float],
) -> None:
    """
    Prepares the DC list attributes for the QDAC channel. This function assumes the use of the
    Qdac2 driver from qcodes_contrib_drivers. This also assumes that the VoltageGate objects have
    their QdacSpec objects configured with the qdac_output_port and opx_trigger_out.
    """
    virtual_dc_set = node.machine.virtual_dc_sets[virtual_dc_set_id]
    physical_dc_lists = _find_physical_dc_lists(virtual_dc_set, axis_name, axis_values)

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

    for name, voltages in physical_dc_lists.items():
        dc_list = node.machine.qdac.channel(virtual_dc_set.channels[name].qdac_spec.qdac_output_port).dc_list(
            voltages=voltages,
            dwell_s=node.parameters.qdac_dwell_time_us / 1e6,
            stepped=True,
        )
        dc_list.start_on_external(trigger=trigger)
