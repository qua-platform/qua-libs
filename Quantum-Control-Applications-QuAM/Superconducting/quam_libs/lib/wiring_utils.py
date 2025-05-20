import os
import json
import quam_libs.lib.QuAM as QuAM


def create_controller_to_qubit_mapping(wiring : dict[dict] = None) -> dict:
    """
    Creates a mapping from controller/port notation to qubit line names.
    
    Args:
        wiring: The wiring dictionary from the QuAM machine
        
    Returns:
        A dictionary mapping controller/port notation to qubit line names
    """
    
    if wiring is None:
        machine = QuAM.load()
        wiring = machine.wiring
        
    mapping = {}
    
    # Iterate through qubits in the wiring
    for qubit_id, qubit_wiring in wiring.get('qubits', {}).items():
        # Get the line types (resonator, drive, flux)
        for line_type, ports in qubit_wiring.items():
            # Get the port reference
            for port_key, port_ref in ports.items():
                if 'opx_output' == port_key or 'opx_input' == port_key:
                    # Extract controller and port from the reference
                    # Example reference: "#/ports/analog_outputs/con1/1/1"
                    
                    controller = port_ref.controller_id
                    fem_id = port_ref.fem_id
                    port = port_ref.port_id
                    # Create the controller/port notation
                    con_port = f"{controller}/{fem_id}/{port}"
                    
                    # Create the qubit line name based on line type
                    if line_type == 'rr':
                        qubit_line = f"{qubit_id[-2]}-rr" # [f"{qubit_id}-rr-{port_key.split('_')[-1]}"]
                    elif line_type == 'xy':
                        qubit_line = f"{qubit_id}-xy"
                    elif line_type == 'z':
                        qubit_line = f"{qubit_id}-z"
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                    
                    if line_type == 'rr' and con_port in mapping and mapping[con_port] != qubit_line:
                        print(f" Warning : Multiple qubit lines for {con_port}: {mapping[con_port]} and {qubit_line}")            
                    
                    mapping[con_port] = qubit_line
    
    return mapping

if __name__ == "__main__":
   
    # Test the mapping function
    mapping = create_controller_to_qubit_mapping()
    
    # Print the results
    print("Controller/Port to Qubit Line Mapping:")
    for con_port, qubit_line in mapping.items():
        print(f"{con_port} -> {qubit_line}")