import numpy as np
from typing import Dict

from qualibrate.core import QualibrationNode

__all__ = [
    "get_axis_names_and_validate",
    "get_voltage_arrays",
    "set_dac_offsets",
]

def get_axis_names_and_validate(node: QualibrationNode): 
    """In the case that x_axis_name or y_axis_name is None, assign the first and second elements of QDs."""

    quantum_dots = list(node.machine.quantum_dots.keys())
    x_axis_name = node.parameters.x_axis_name
    y_axis_name = node.parameters.y_axis_name

    if node.parameters.x_axis_name is None:
        x_axis_name = quantum_dots[0]
    
    if node.parameters.y_axis_name is None: 
        y_axis_name = quantum_dots[1]

    x_obj = node.machine.get_component(x_axis_name)
    y_obj = node.machine.get_component(y_axis_name)

    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    node.namespace["axes_names"] = {"x_axis": x_axis_name, "y_axis": y_axis_name, "gate_set_id" : vgs_id} 

    return x_axis_name, y_axis_name, vgs_id

def get_voltage_arrays(node: QualibrationNode):
    """
    Get the voltage arrays that are to be outputted via the OPX. 

    This depends on if node.parameters.dc_control is True or False. 
        - True: The offset is applied via the DC controller, and the OPX sweep is centred around the OPX's effective 0. 
        - False: The offset is applied via the OPX, and the DC controller is not even connected. 
    """

    x_span = node.parameters.x_span
    x_points = node.parameters.x_points
    y_span = node.parameters.y_span
    y_points = node.parameters.y_points

    x_offset = node.parameters.x_offset
    y_offset = node.parameters.y_offset

    x_volts = np.linspace(-x_span/2, x_span/2, x_points)
    y_volts = np.linspace(-y_span/2, y_span/2, y_points)

    # DC control: offsets will be via the DAC
    # Non-DC Control: offsets will be via the OPX
    if not node.parameters.dc_control: 
        if x_offset is not None: 
            if x_offset > 2.5: 
                raise ValueError(f"X offset greater than OPX output limit of 2.5. Requested {x_offset}")
            x_volts = x_volts + x_offset
        if y_offset is not None: 
            if y_offset > 2.5: 
                raise ValueError(f"Y offset greater than OPX output limit of 2.5. Requested {y_offset}")
            y_volts = y_volts + y_offset
    
    return x_volts, y_volts


def set_dac_offsets(node: QualibrationNode, dc_set_id: str, voltages: Dict[str, float | None]) -> None: 
    """Sequentially apply the DC offsets to the DAC. This function should be run after running qmm.connect(skip_dacs = False)"""
    dc_set = node.machine.virtual_dc_sets[dc_set_id]
    

    # If the supplied offset value is None, then default to the already applying value
    for gate_name, voltage in voltages.items(): 
        if voltage is None: 
            current_value = dc_set.get_voltage(gate_name, requery = True)
            voltages[gate_name] = current_value
        
        node.log(
            f"Setting DC Voltages via VirtualDCSet. {gate_name} : {voltages[gate_name]}V"
        )
        
    dc_set.set_voltages(voltages)
