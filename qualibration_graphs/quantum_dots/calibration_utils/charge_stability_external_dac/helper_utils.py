import numpy as np
from typing import Sequence

from qualibrate.core import QualibrationNode


__all__ = [
    "get_voltage_arrays",
    "validate_opx_limit",
]


def get_voltage_arrays(node: QualibrationNode):
    """Build the fast OPX sweep and slow external-DAC sweep arrays."""
    opx_offset = node.parameters.opx_offset or 0.0
    opx_array = np.linspace(
        -node.parameters.opx_span / 2,
        node.parameters.opx_span / 2,
        node.parameters.opx_points,
    ) + opx_offset

    dac_array = np.linspace(
        -node.parameters.dac_span / 2,
        node.parameters.dac_span / 2,
        node.parameters.dac_points,
    )

    opx_axis_name = node.parameters.opx_fast_axis_name
    dac_axis_name = node.parameters.dac_slow_axis_name
    opx_sweep_object = node.machine.get_component(opx_axis_name)
    vgs_id = opx_sweep_object.voltage_sequence.gate_set.name

    node.namespace["axes_names"] = {
        "x_axis": opx_axis_name,
        "y_axis": dac_axis_name,
        "gate_set_id": vgs_id,
    }
    node.namespace["dac_values"] = dac_array

    return opx_array, dac_array, vgs_id


def validate_opx_limit(
    node: QualibrationNode,
    axis_name: str,
    array: Sequence[float],
    limit: float = 2.5,
):
    """Validate that the resolved physical OPX sweep stays within hardware limits."""
    vgs_id = node.namespace["axes_names"]["gate_set_id"]
    vgs = node.machine.virtual_gate_sets[vgs_id]

    max_value = float(np.max(np.abs(array)))
    max_physical = vgs.resolve_voltages({axis_name: max_value})
    max_resolved = float(np.max(np.abs(list(max_physical.values()))))
    if max_resolved > limit:
        raise ValueError(
            f"Resolved OPX sweep for {axis_name} exceeds the {limit} V limit ({max_resolved} V)."
        )

