"""
File containing the definition of custom random single qubit gates for XEB, used as
a gate set for the XEB benchmarking. We provide here the definition of two gate sets,
based on two different papers implementing the XEB benchmarking protocol.
Note that the gates are assumed to be all doable through the modulation of the amplitude matrix of a baseline
calibrated X/2 (SX) pulse. One could modify the XEB code to allow for dedicated pulses for each gate (using the
QUA switch-case functionality).

The first gate set is based on the Google Supremacy paper (https://www.nature.com/articles/s41586-019-1666-5),
where the gate set consists of the X/2, Y/2 and SW gates. The SW gate is defined as the unitary gate corresponding
to the square root of the W gate, where W is a pi/2 rotation around the X+Y axis.

The second gate set is based on https://arxiv.org/abs/1608.00263, where the gate set consists of the X/2, Y/2
and T gates. T is a pi/4 rotation around the Z axis and is implemented as a virtual frame rotation in the QUA code.
We define an amplitude matrix for the latter gate for consistency with the other gates.

Author: Arthur Strauss - Quantum Machines
Last updated: 2024-04-30
"""

from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from qiskit.circuit.library import UnitaryGate, RYGate
from scipy.linalg import sqrtm
import numpy as np


def generate_gate_set(gate_set: str):
    """
    Generate a gate set from a string for random single qubit gates for XEB.
    Two of the gates are always X/2 and Y/2, the third gate is determined by the provided string
    (for now, can be only "sw" or "t")
    Each single qubit gate is assumed to be playable through the modulation of the amplitude matrix of a baseline
    calibrated X/2 (SX) pulse
    Args:
        gate_set (str): String of gate names separated by commas
    Returns:
        dict: Dictionary of single qubit gates
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
        raise ValueError("Invalid gate set")

    return gate_dict
