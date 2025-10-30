from qiskit import QuantumCircuit
import numpy as np
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from calibration_utils.two_qubit_interleaved_rb.rb_utils import EPS

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


def layerize_quantum_circuit(qc: QuantumCircuit) -> QuantumCircuit:

    dag = circuit_to_dag(qc)
    layered_qc = QuantumCircuit(qc.num_qubits)
    for layer in dag.layers():
        layer_as_circuit = dag_to_circuit(layer['graph'])
        layer_as_circuit.barrier()
        layered_qc = layered_qc.compose(layer_as_circuit)

    return layered_qc


def collect_gates_as_2_qubit_layers(qc: QuantumCircuit) -> list[str]:
    """
    Iterates over a qiskit quantum circuit and collects the gates as strings.
    The circuit is separated by barriers. Between each barrier, the function collects the gates as strings.
    The name of the gate as string should be like <name>_<qubit>. Between gates place a dash.

    Args:
        qc (QuantumCircuit): The quantum circuit to iterate over.

    Returns:
        list[str]: A list of strings representing the gates in the circuit.
    """
    gate_strings = []
    current_gates = []

    for instruction in qc:
        if instruction.name == 'barrier':
            if current_gates:
                gate_strings.append('-'.join(current_gates))
                current_gates = []
        else:
            for qubit in instruction.qubits:
                gate_str = f"{instruction.name}_{qubit.index}"
                current_gates.append(gate_str)

    if current_gates:
        gate_strings.append('-'.join(current_gates))

    return gate_strings


def quantum_circuit_to_integer_and_angle_list(qc: QuantumCircuit, gate_map: dict, ignore_barriers: bool = True) -> tuple[list[int], list[float]]:

    gate_interger_list = []
    rz_angle_count_q0 = []
    rz_angle_count_q1 = []
    qubit_index_list = [] # need to make this more robust

    layered_qc = layerize_quantum_circuit(qc)

    ang_q0 = 0
    ang_q1 = 0

    for instruction in layered_qc:

        qubit_index = instruction.qubits[0]._index # TODO : make this more robust

        if instruction.name == 'rz':
            # only follows the accumulation of rz angles. does not add a to gate list
            if qubit_index == 0:
                ang_q0 += instruction.params[0]
            elif qubit_index == 1:
                ang_q1 += instruction.params[0]
            else:
                raise ValueError("Only 2 qubits are supported")

        else:
            # add a gate
            if instruction.name in ['rx', 'ry']:
                axis = instruction.name[1]
                theta = instruction.params[0]
                if (theta/np.pi - 0.5) < EPS:
                    gate_int = gate_map[f's{axis}']
                elif (theta/np.pi - (-0.5)) < EPS:
                    gate_int = gate_map[f'{axis}270']
                elif (theta/np.pi - 1) < EPS or (theta/np.pi - (-1)) < EPS:
                    gate_int = gate_map[f'{axis}']
                else:
                    raise ValueError(f"Only 90, -90, 180, -180 rotations are supported not angle={theta}")

            elif instruction.name == 'barrier' and ignore_barriers:
                continue

            else:
                gate_int = gate_map[instruction.name]

            gate_interger_list.append(gate_int)
            qubit_index_list.append(qubit_index)
            rz_angle_count_q0.append(ang_q0)
            rz_angle_count_q1.append(ang_q1)

            ang_q0 = 0
            ang_q1 = 0


    # measurement
    gate_interger_list.append(gate_map['measure'])
    rz_angle_count_q0.append(0)
    rz_angle_count_q1.append(0)
    qubit_index_list.append(0)

    return gate_interger_list, qubit_index_list, rz_angle_count_q0, rz_angle_count_q1