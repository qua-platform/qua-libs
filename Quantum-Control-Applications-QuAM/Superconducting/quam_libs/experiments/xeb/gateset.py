from qiskit.circuit.library import RYGate
from .qua_gate import QUAGate
from qm.qua import amp
import numpy as np
from typing import Literal, Callable, Optional, List, Dict, Union
from quam_libs.components import Transmon


def play_sq_gate_macro(baseline_gate_name: str, amp_matrix: Optional[List] = None) -> Callable[[Transmon], None]:
    """
    Play a single qubit gate on a given qubit element.

    This function plays a single qubit gate on a given qubit element, by modulating the amplitude matrix of a baseline
    calibrated X/2 (SX) pulse.

    Args:
        baseline_gate_name (str): Name of the baseline gate implementing a pi/2 rotation around the x-axis.
        amp_matrix (np.ndarray): Amplitude matrix of the gate.
    """

    def play_sq_gate(qubit: Transmon):
        qubit.xy.play(
            baseline_gate_name,
            amplitude_scale=amp(*amp_matrix) if amp_matrix is not None else None,
        )

    return play_sq_gate


def play_virtual_t_gate(qubit: Transmon):
    """
    Play a virtual T gate on a given qubit element.

    This function plays a virtual T gate on a given qubit element, by doing a frame rotation of pi/8 around the Z axis.
    """
    qubit.xy.frame_rotation_2pi(0.125)


class QUAGateSet(dict):
    """
    Class to store a set of QUA gates for XEB.

    Args:
        gate_set (Union[Literal['sw', 't', Dict[int, QUAGate]]): Gate Set for XEB. Two native gates sets are available
            through the strings 'sw' and 't'. Those sets respectively contain the X/2, Y/2, and SW gates, and the X/2, Y/2,
            and T gates. In the "sw" gate set, all gates are assumed to be playable through the modulation of the amplitude
            matrix of a baseline calibrated X/2 (SX) pulse.
            In the "t" gate set, the T gate is a virtual pi/4 rotation around the Z axis.

            Alternatively, a custom gate set can be provided as a dictionary of QUAGate objects. If a custom gate
            set is provided, the dictionary keys should be the gate indices. The gate indices should start from 0 and be
            consecutive integers. The dictionary values should be QUAGate objects.
        baseline_gate_name (str): Name of the baseline gate implementing a pi/2 rotation around the x-axis.
    """

    def __init__(
        self,
        gate_set: Union[Literal["sw", "t"], Dict[int, QUAGate]],
        baseline_gate_name: Optional[str] = None,
    ):
        if isinstance(gate_set, dict):
            for key, value in gate_set.items():
                if not isinstance(key, int) or key < 0:
                    raise ValueError("Invalid gate set: keys should be positive integers.")
                if not isinstance(value, QUAGate):
                    raise ValueError("Invalid gate set: values should be QUAGate objects.")
            max_key = max(gate_set.keys())
            if max_key != len(gate_set) - 1:
                raise ValueError("Invalid gate set: keys should be consecutive integers starting from 0.")
            for key in range(max_key + 1):
                if key not in gate_set:
                    raise ValueError(f"Invalid gate set: missing gate with index {key}.")
            super().__init__(gate_set)
        else:
            if baseline_gate_name is None:
                raise ValueError("Baseline gate name must be provided for predefined gate sets.")
            super().__init__(generate_gate_set(gate_set, baseline_gate_name))
        self.name = gate_set

    def __str__(self):
        return f"QUAGateSet({super().__str__()})"

    @property
    def run_through_amp_matrix_modulation(self):
        """
        Check if all gates in the set can be played through the modulation of the amplitude matrix of a baseline gate.
        """
        return all(gate.amp_matrix is not None for gate in self.values())


def generate_gate_set(gate_set: Literal["sw", "t"], baseline_gate_name: str) -> Dict[int, QUAGate]:
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
        baseline_gate_name (str): Name of the baseline gate implementing a pi/2 rotation around the x-axis.

    Returns:
        dict: Dictionary containing the generated single qubit gates.
    Raises:
        ValueError: If an invalid gate set string is provided.
    """

    sx_gate = QUAGate("sx", play_sq_gate_macro(baseline_gate_name), amp_matrix=[1.0, 0.0, 0.0, 1.0])
    sy = RYGate(np.pi / 2).to_matrix()
    sy_gate = QUAGate(
        ("sy", sy),
        play_sq_gate_macro(baseline_gate_name, [0.0, -1.0, 1.0, 0.0]),
        amp_matrix=[0.0, -1.0, 1.0, 0.0],
    )

    gate_dict = {0: sx_gate, 1: sy_gate}
    if gate_set == "sw":
        sw_amp_matrix = list(0.70710678 * np.array([1.0, -1.0, 1.0, 1.0]))
        sw_gate = QUAGate(
            ("sw", np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]]) / np.sqrt(2)),
            play_sq_gate_macro(baseline_gate_name, sw_amp_matrix),
            amp_matrix=sw_amp_matrix,
        )
        gate_dict[2] = sw_gate
    elif gate_set == "t":
        t_gate = QUAGate("t", play_virtual_t_gate)
        gate_dict[2] = t_gate
    else:
        raise ValueError(f"Invalid gate set: {gate_set}. Allowed values are 'sw' or 't'.")

    return gate_dict
