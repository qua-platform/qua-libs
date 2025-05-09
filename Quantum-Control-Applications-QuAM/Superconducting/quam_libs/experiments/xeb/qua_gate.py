from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
from qiskit.circuit.library import UnitaryGate
import numpy as np
from typing import Callable, Union, Optional, Tuple, Any, List
import warnings
from quam.components import Channel
from quam_libs.components import Transmon, TransmonPair


def _validate_amp_matrix(amp_matrix: Optional[List] = None):
    """
    Validate the amplitude matrix of the gate.

    Args:
        amp_matrix: List: The amplitude matrix of the gate.
    """
    if amp_matrix is not None:
        if not isinstance(amp_matrix, List):
            raise ValueError("Amplitude matrix must be a list.")
        if len(amp_matrix) != 4:
            raise ValueError("Amplitude matrix must have 4 elements.")
        if not all([isinstance(amp, (float, int)) for amp in amp_matrix]):
            raise ValueError("Amplitude matrix elements must be of type float or int.")

    return amp_matrix


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
            of a baseline X/2 gate, this parameter should contain the amplitude matrix enabling the computation of the
            gate from a SX pulse. This avoids the need to provide a gate_macro and the induced switch case latency
            overhead. Note: The removal of the switch case overhead is possible only if all gates in the gate set
            do have this attribute set.
    """

    def __init__(
        self,
        gate: Union[str, Tuple[str, np.ndarray], Gate],
        gate_macro: Optional[Callable[[Union[Transmon, TransmonPair]], None]] = None,
        amp_matrix: Optional[List] = None,
    ):
        self.gate_macro = gate_macro
        self.amp_matrix = _validate_amp_matrix(amp_matrix)
        if gate_macro is None and amp_matrix is None:
            warnings.warn("No gate macro or amp_matrix provided, the gate will not be implemented in QUA.")

        if isinstance(gate, str):
            gate = gate.lower()
            if gate == "cnot":
                self._gate = gate_map()["cx"]
            else:
                try:
                    self._gate = gate_map()[gate]
                except KeyError:
                    raise ValueError(f"Invalid gate: {gate}, please specify it through its matrix definition.")
        elif isinstance(gate, Tuple):
            if not isinstance(gate[1], np.ndarray):
                raise ValueError("Invalid gate definition, the matrix should be a numpy array.")
            if not isinstance(gate[0], str):
                raise ValueError("Invalid gate definition, the name should be a string.")
            self._gate = UnitaryGate(gate[1], label=gate[0])
            
        elif isinstance(gate, Gate):
            self._gate = gate
            
        else:
            raise ValueError(
                "Invalid gate definition, please provide a valid gate name"
                " or a tuple containing custom name and matrix definition."
            )

    def __str__(self):
        return f"QUAGate({self.gate.name})"

    def __repr__(self):
        return f"QUAGate({self.gate.name})"

    @property
    def name(self):
        return self.gate.name if self.gate.label is None else self.gate.label

    @property
    def gate(self) -> Gate:
        return self._gate
