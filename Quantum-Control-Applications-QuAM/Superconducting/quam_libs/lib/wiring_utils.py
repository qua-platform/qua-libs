from quam_libs.components import QuAM


def create_controller_to_qubit_mapping(wiring : dict[dict] = None) -> dict:
    """
    Creates a mapping from controller/port notation to qubit line names, sorted by line types (rr, xy, z).
    
    Args:
        wiring: The wiring dictionary from the QuAM machine
        
    Returns:
        A dictionary mapping controller/port notation to qubit line names, organized by line types
    """
    
    if wiring is None:
        machine = QuAM.load()
        wiring = machine.wiring
        
    # Initialize dictionaries for each line type
    rr_mapping = {}
    xy_mapping = {}
    z_mapping = {}
    
    # Iterate through qubits in the wiring
    for qubit_id, qubit_wiring in wiring.get('qubits', {}).items():
        # Get the line types (resonator, drive, flux)
        for line_type, ports in qubit_wiring.items():
            # Get the port reference
            for port_key, port_ref in ports.items():
                if 'opx_output' == port_key or 'opx_input' == port_key:
                    # Extract controller and port from the reference
                    # Example reference: "#/ports/analog_outputs/con1/1/1"
                    
                    if type(port_ref) == str:
                        controller = port_ref.split('/')[-3]
                        fem_id = port_ref.split('/')[-2]
                        port = port_ref.split('/')[-1]
                    else:
                        controller = port_ref.controller_id
                        fem_id = port_ref.fem_id
                        port = port_ref.port_id
                    # Create the controller/port notation
                    con_port = f"{controller}/{fem_id}/{port}"
                    
                    # Create the qubit line name based on line type
                    if line_type == 'rr':
                        qubit_line = f"{qubit_id[-2]}-rr" # [f"{qubit_id}-rr-{port_key.split('_')[-1]}"]
                        if con_port in rr_mapping and rr_mapping[con_port] != qubit_line:
                            print(f" Warning : Multiple qubit lines for {con_port}: {rr_mapping[con_port]} and {qubit_line}")
                        rr_mapping[con_port] = qubit_line
                    elif line_type == 'xy':
                        qubit_line = f"{qubit_id}-xy"
                        xy_mapping[con_port] = qubit_line
                    elif line_type == 'z':
                        qubit_line = f"{qubit_id}-z"
                        z_mapping[con_port] = qubit_line
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
    
    # Combine all mappings in the desired order
    sorted_mapping = {
        'rr': rr_mapping,
        'xy': xy_mapping,
        'z': z_mapping
    }
    
    return sorted_mapping

if __name__ == "__main__":
   
    # Test the mapping function
    mapping = create_controller_to_qubit_mapping()
    
    # Print the results
    print("Controller/Port to Qubit Line Mapping (sorted by line type):")
    for line_type, type_mapping in mapping.items():
        print(f"\n{line_type.upper()} lines:")
        for con_port, qubit_line in type_mapping.items():
            print(f"{con_port} -> {qubit_line}")