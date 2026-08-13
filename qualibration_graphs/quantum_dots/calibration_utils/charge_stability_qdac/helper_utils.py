import numpy as np
from typing import Dict

from qualibrate.core import QualibrationNode

__all__ = [
    "get_voltage_arrays",
    "set_dac_offsets",
]

def axis_source_bools(node: QualibrationNode): 
    return node.parameters.x_from_qdac, node.parameters.y_from_qdac

def get_voltage_arrays(node: QualibrationNode):
    """
    Get the voltage arrays that are to be outputted via the OPX.

    This depends on if whether X or Y axis is sourced from the QDAC. If not, then the offset (if not None) will be applied to the sweep axis from the OPX. 
    """

    x_ext, y_ext = axis_source_bools(node)

    # First construct the non-offset arrays
    x_span = node.parameters.x_span
    x_points = node.parameters.x_points
    y_span = node.parameters.y_span
    y_points = node.parameters.y_points

    x_offset = node.parameters.x_offset
    y_offset = node.parameters.y_offset

    x_volts = np.linspace(-x_span / 2, x_span / 2, x_points)
    y_volts = np.linspace(-y_span / 2, y_span / 2, y_points)

    if not x_ext: # X offset to come from the OPX
        if x_offset is not None:
            if abs(x_offset) > 2.5:
                raise ValueError(f"X offset greater than OPX output limit of 2.5. Requested {x_offset}")
            x_volts = x_volts + x_offset
    if not y_ext: # Y offset to come from the OPX
        if y_offset is not None:
            if abs(y_offset) > 2.5:
                raise ValueError(f"Y offset greater than OPX output limit of 2.5. Requested {y_offset}")
            y_volts = y_volts + y_offset

    return x_volts, y_volts


def set_dac_offsets(node: QualibrationNode, dc_set_id: str, voltages: Dict[str, float | None]) -> None:
    """Sequentially apply the DC offsets to the DAC. This function should be run after running qmm.connect(skip_dacs = False)"""
    dc_set = node.machine.virtual_dc_sets[dc_set_id]

    # If the supplied offset value is None, then default to the already applying value
    for gate_name, voltage in voltages.items():
        if voltage is None:
            current_value = dc_set.get_voltage(gate_name, requery=True)
            voltages[gate_name] = current_value

        node.log(f"Setting DC Voltages via VirtualDCSet. {gate_name} : {voltages[gate_name]}V")

    dc_set.set_voltages(voltages)
