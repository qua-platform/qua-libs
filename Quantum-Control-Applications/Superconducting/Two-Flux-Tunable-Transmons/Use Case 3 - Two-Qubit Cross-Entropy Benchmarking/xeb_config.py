from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np


@dataclass
class XEBConfig:
    """
    Configuration for running XEB

    Args:
        seqs (int): Number of random sequences to run per depth
        depths (np.ndarray): Array of depths to iterate through
        n_shots (int): Number of averages per sequence
        impose_0_cycle (bool): Whether to impose the first gate at 0-cycle
        apply_two_qb_gate (bool): Whether to apply a two-qubit gate or not
        gate_set_choice (str): Choice of gate set for XEB (choose "sw" or "t")
        save_dir (str): Directory where the data will be saved
        should_save_data (bool): Whether to save the data
        generate_new_data (bool): Whether to generate new data


    """

    seqs: int
    depths: np.ndarray
    n_shots: int = 101
    impose_0_cycle: bool = False
    apply_two_qb_gate: bool = False
    gate_set_choice: Literal["sw", "t"] = "sw"
    save_dir: str = ""
    should_save_data: bool = True
    generate_new_data: bool = True
