from qiskit import QuantumCircuit
import numpy as np
from typing import List, Tuple, Dict, Union

# Gate to integer mapping for single qubit gates
SINGLE_QUBIT_GATE_MAP = {

    'sx': 0,
    'x': 1,
    'sy': 2,
    'y': 3,
    'rz(pi/2)': 4,
    'rz(pi)': 5,
    'rz(3pi/2)': 6,
    'idle': 7
    
}

# Two-qubit gate mapping
TWO_QUBIT_GATE_MAP = {
    'cz': 64,  # Start at 64 to avoid overlap with single-qubit combinations
    'idle_2q': 65
}

def get_gate_name(gate) -> str:
    """Extract the gate name and parameters in a standardized format."""
    name = gate.name.lower()
    if name == 'rz':
        angle = gate.params[0]
        if np.isclose(angle, np.pi/2):
            return name + '(pi/2)'
        elif np.isclose(angle, np.pi) or np.isclose(angle, -np.pi):
            return name + '(pi)'
        elif np.isclose(angle, 3*np.pi/2) or np.isclose(angle, -np.pi/2):
            return name + '(3pi/2)'
        else:
            raise ValueError(f"Unsupported angle: {angle}")
    return name

def get_layer_integer(layer: Union[list[int, int], str]) -> int:
    """
    Convert a layer configuration to an integer.
    
    Args:
        layer: Either a tuple of two single-qubit gate integers or a two-qubit gate name
        
    Returns:
        Integer representing the layer configuration
    """
    if isinstance(layer[0], list):
        # For single-qubit gate pairs, use a formula to generate unique integers
        # This maps (g1, g2) to a unique integer in range [0, 63]
        return layer[0][0] * len(SINGLE_QUBIT_GATE_MAP) + layer[0][1]
    elif isinstance(layer[0], str):
        return TWO_QUBIT_GATE_MAP[layer[0]]
    else:
        raise ValueError(f"Unsupported layer : {layer[0]}")

def process_circuit_to_integers(circuit: QuantumCircuit) -> List[int]:
    """
    Process a quantum circuit and return a list of integers representing the circuit.
    
    Args:
        circuit: A Qiskit QuantumCircuit with 2 qubits
        
    Returns:
        List of integers representing the circuit configuration between barriers
    """
    result = []
    current_layer = [[7,7]]  # Initialize with idle gates for both qubits
    
    for instruction in circuit:
        if instruction.operation.name == 'barrier' and current_layer != [[7,7]]:
            # Convert current layer to integer and add to result
            layer_int = get_layer_integer(current_layer)
            result.append(layer_int)
            current_layer = [[7,7]]  # Reset to idle gates
            continue
            
        gate_name = get_gate_name(instruction.operation)
        
        if gate_name == 'cz' or gate_name == 'idle_2q':
            # If we have a CZ or idle_2q gate, it's a two-qubit operation
            if current_layer != [[7,7]]:
                raise ValueError(f"{gate_name} gate found with single qubit gates in the layer")
            else:
                current_layer = [gate_name]
                # # If there are any single-qubit gates, process them first
                # layer_int = get_layer_integer(tuple(current_layer))
                # result.append(layer_int)
                # current_layer = []
        elif gate_name in SINGLE_QUBIT_GATE_MAP:
            # Single-qubit gate
            qubit_index = instruction.qubits[0]._index
            current_layer[0][qubit_index] = SINGLE_QUBIT_GATE_MAP[gate_name]
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")  
    
    # Process any remaining gates after the last barrier
    if current_layer != [[7,7]]:
        layer_int = get_layer_integer(current_layer)
        result.append(layer_int)
    
    return result