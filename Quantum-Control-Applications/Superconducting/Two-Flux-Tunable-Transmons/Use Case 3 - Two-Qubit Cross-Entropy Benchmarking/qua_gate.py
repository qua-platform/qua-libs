from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from qiskit.circuit.library import UnitaryGate
from dataclasses import dataclass
import numpy as np
from typing import Callable, Union, Optional
from quam.examples.superconducting_qubits.components import Transmon
import warnings


class QUAGate:
    """
    Class to store a QUA gate, containing the logical definition of the gate and its QUA implementation through
    a QUA macro.

    Args:
        gate: str|np.ndarray: The logical definition of the gate, either a string with the name of the gate or a
            numpy array with the gate matrix.
        gate_macro: Callable: The QUA macro that implements the gate.
    """

    def __init__(self, gate: Union[str, np.ndarray], gate_macro: Optional[Callable[[Transmon, Transmon], None]] = None):
        self.gate_macro = gate_macro
        if gate_macro is None:
            warnings.warn("No gate macro provided, the gate will not be implemented in QUA.")

        if isinstance(gate, str):
            gate = gate.lower()
            if gate == "cnot":
                self.gate = gate_map()["cx"]
            else:
                try:
                    self.gate = gate_map()[gate]
                except KeyError:
                    raise ValueError(f"Invalid gate: {gate}, please specify it through its matrix definition.")
        elif isinstance(gate, np.ndarray):
            self.gate = UnitaryGate(gate, label="custom_2q_gate")
        else:
            raise ValueError("Invalid gate definition, please provide a valid gate matrix or name.")
