from dataclasses import dataclass, asdict, field
from typing import Literal, List, Union, Optional

import numpy as np
from gateset import generate_gate_set
from qua_gate import QUAGate


@dataclass
class XEBConfig:
    """
    Experiments parameters for running XEB

    Args:
        seqs (int): Number of random sequences to run per depth
        depths (np.ndarray): Array of depths to iterate through
        n_shots (int): Number of averages per sequence
        qubits_ids (List[Union[int, str]]): List of qubit identifiers for the targeted qubits as specified in the QuAM
        baseline_gate_name (str): Name of the baseline gate implementing a pi/2 rotation around the x-axis (default "sx")
        gate_set_choice (str): Choice of gate set for XEB (choose "sw" or "t")
        two_qb_gate (Optional[QUAGate]): Two-qubit gate to be used in the experiment
        impose_0_cycle (bool): Whether to impose the first gate at 0-cycle
        save_dir (str): Directory where the data will be saved
        should_save_data (bool): Whether to save the data
        generate_new_data (bool): Whether to generate new data
        disjoint_processing (bool): Whether to process the data in a disjoint manner (that is compute qubit states independently, relevant only when no two-qubit gate is provided)


    """

    seqs: int
    depths: Union[np.ndarray, List[int]]
    n_shots: int = 101
    qubits_ids: List[Union[int, str]] = field(default_factory=lambda: [0, 1])
    baseline_gate_name: str = "sx"
    gate_set_choice: Literal["sw", "t"] = "sw"
    two_qb_gate: Optional[QUAGate] = None
    impose_0_cycle: bool = False
    save_dir: str = ""
    should_save_data: bool = True
    generate_new_data: bool = True
    disjoint_processing: bool = False

    def __post_init__(self):
        if isinstance(self.depths, List):
            self.depths = np.array(self.depths)

        self.gate_dict = generate_gate_set(self.gate_set_choice)
        self.n_qubits = len(self.qubits_ids)
        self.dim = 2**self.n_qubits
