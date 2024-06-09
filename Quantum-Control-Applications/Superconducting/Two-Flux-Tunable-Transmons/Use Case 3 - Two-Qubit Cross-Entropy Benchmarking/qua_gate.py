from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from qiskit.circuit.library import UnitaryGate
import numpy as np
from typing import Callable, Union, Optional, Tuple, Any, List
import warnings


class QUAGate:
    """
    Class to store a QUA gate, containing the logical definition of the gate and its QUA implementation through
    a QUA macro.

    Args:
        gate: str|(str, np.ndarray): The logical definition of the gate, either a string with the name of the gate or a
        tuple containing the gate name and a numpy array defining its matrix. All standard gates present in Qiskit
        standard library are supported with the string definition.
        gate_macro: Callable: The QUA macro that implements the gate.
        amp_matrix: List: If the gate is a single qubit gate and can be obtained by a modulation of the amplitude matrix
        of a baseline gate, this parameter should contain the amplitude matrix of the gate. This avoids the need to
        provide a gate_macro and the induced switch case latency overhead.
    """

    def __init__(self, gate: Union[str, Tuple[str, np.ndarray]],
                 gate_macro: Optional[Callable[[Any], None]] = None,
                 amp_matrix: Optional[List] = None):
        self.gate_macro = gate_macro
        if amp_matrix is not None:
            assert len(amp_matrix) == 4, "Amplitude matrix must have 4 elements."
        self.amp_matrix = amp_matrix
        if gate_macro is None and amp_matrix is None:
            warnings.warn("No gate macro or amp_matrix provided, the gate will not be implemented in QUA.")

        if isinstance(gate, str):
            gate = gate.lower()
            if gate == "cnot":
                self.gate = gate_map()["cx"]
            else:
                try:
                    self.gate = gate_map()[gate]
                except KeyError:
                    raise ValueError(f"Invalid gate: {gate}, please specify it through its matrix definition.")
        elif isinstance(gate, Tuple):
            self.gate = UnitaryGate(gate[1], label=gate[0])
        else:
            raise ValueError("Invalid gate definition, please provide a valid gate name"
                             " or a tuple containing custom name and matrix definition.")

    def __str__(self):
        return f"QUAGate({self.gate.name})"

    @property
    def name(self):
        return self.gate.name if self.gate.label is None else self.gate.label