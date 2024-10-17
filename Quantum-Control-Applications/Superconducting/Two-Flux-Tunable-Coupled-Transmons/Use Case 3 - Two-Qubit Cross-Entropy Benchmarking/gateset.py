from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from qiskit.circuit.library import UnitaryGate, RYGate
from scipy.linalg import sqrtm
import numpy as np
from typing import Literal


def generate_gate_set(gate_set: Literal["sw", "t"]) -> dict:
    """
    Generate a gate set from a string for random single qubit gates for XEB.

    This function generates a dictionary of single-qubit gates for XEB (Cross-Entropy Benchmarking)
    based on the provided gate set string.

    The available gate sets are:
    - "sw": X/2, Y/2, and SW gates. SW gate is defined as the square root of the W gate.
    - "t": X/2, Y/2, and T gates. T gate is a pi/4 rotation around the Z axis.

    Each single qubit gate is assumed to be playable through the modulation of the amplitude
    matrix of a baseline calibrated X/2 (SX) pulse (motivating the systematic definition of an
    amplitude matrix for each gate). However in the main QUA script, one could choose how to play
    each gate based on a QUA switch-case statement.

    Args:
        gate_set (Literal['sw', 't']): String indicating the desired gate set.

    Returns:
        dict: Dictionary containing the generated single qubit gates.
    Raises:
        ValueError: If an invalid gate set string is provided.
    """

    SX, T, X, Y = [gate_map()[gate] for gate in ["sx", "t", "x", "y"]]
    SY = RYGate(np.pi / 2)

    # Define the relevant amplitude matrices for all single-qubit gates (all generated through initial X90 gate)
    X90_dict = {"gate": SX, "amp_matrix": np.array([1.0, 0.0, 0.0, 1.0])}
    Y90_dict = {"gate": SY, "amp_matrix": np.array([0.0, -1.0, 1.0, 0.0])}
    gate_dict = {0: X90_dict, 1: Y90_dict}
    if gate_set == "sw":
        W = UnitaryGate((X.to_matrix() + Y.to_matrix()) / np.sqrt(2), label="W")
        SW = UnitaryGate(sqrtm(W), label="sw")  # from Supremacy paper
        SW_dict = {"gate": SW, "amp_matrix": 0.70710678 * np.array([1.0, -1.0, 1.0, 1.0])}
        gate_dict[2] = SW_dict
    elif gate_set == "t":
        T_dict = {
            "gate": T,
            "amp_matrix": np.array([1.0, 0.0, 0.0, 1]),
        }  # No actual need for amp_matrix (done with frame rotation)
        gate_dict[2] = T_dict
    else:
        raise ValueError(f"Invalid gate set: {gate_set}. Allowed values are 'sw' or 't'.")

    return gate_dict
