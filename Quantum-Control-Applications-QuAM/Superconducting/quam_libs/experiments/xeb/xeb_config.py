from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional, Dict

import numpy as np
from qiskit.transpiler import CouplingMap

from .gateset import QUAGateSet
from .qua_gate import QUAGate
from ...components import Transmon, TransmonPair, QuAM
from .macros import get_parallel_gate_combinations as gate_combinations


@dataclass
class XEBConfig:
    """
    Experiments parameters for running XEB

    Args:
        seqs (int): Number of random sequences to run per depth
        depths (np.ndarray): Array of depths to iterate through
        n_shots (int): Number of averages per sequence
        qubits (List[Transmon]): List of active qubits to be used in the experiment
        readout_qubits (Optional[List[Transmon]]): List of active qubits to be used for readout (relevant for multiplexed readout)
        baseline_gate_name (str): Name of the baseline gate implementing a pi/2 rotation around the x-axis (default "sx")
        gate_set_choice (Union[str, Dict[int, QUAGate]]): Choice of gate set for XEB (choose "sw" or "t") or a custom gate set as a dictionary of QUAGate objects
        two_qb_gate (Optional[QUAGate]): Two-qubit gate to be used in the experiment
        qubit_pairs (List[TransmonPair]): List of active qubit pairs to be used in the experiment (for now only one pair is supported)
        readout_pulse_name (str): Name of the readout pulse, should be common to all resonators and should match the name in QuAM (default "readout")
        reset_method (str): Method used to reset the qubits (choose "active" or "cooldown")
        reset_kwargs (Optional[Dict[str, Union[float, str, int]]]): Keyword arguments for the reset method (default {"cooldown_time": 20, "max_tries": None, "pi_pulse_name": None})
        save_dir (str): Directory where the data will be saved
        should_save_data (bool): Whether to save the data
        generate_new_data (bool): Whether to generate new data
        disjoint_processing (bool): Whether to process the data in a disjoint manner (that is compute qubit states independently, relevant only when no two-qubit gate is provided)
        seed (int): Seed for the random number generator


    """

    seqs: int
    depths: Union[np.ndarray, List[int]]
    n_shots: int
    qubits: List[Transmon]
    readout_qubits: Optional[List[Transmon]] = None
    baseline_gate_name: str = "sx"
    gate_set_choice: Union[Literal["sw", "t"], Dict[int, QUAGate]] = "sw"
    two_qb_gate: Optional[QUAGate] = None
    qubit_pairs: Optional[List[TransmonPair]] = field(default_factory=lambda: [])
    readout_pulse_name: str = "readout"
    reset_method: Literal["active", "cooldown"] = "cooldown"
    reset_kwargs: Optional[Dict[str, Union[float, str, int]]] = field(
        default_factory=lambda: {
            "cooldown_time": 20,
            "max_tries": None,
            "pi_pulse": None,
        }
    )
    save_dir: str = ""
    should_save_data: bool = True
    data_folder_name: Optional[str] = None
    generate_new_data: bool = True
    disjoint_processing: bool = False
    seed: int = 1234

    def __post_init__(self):
        if isinstance(self.depths, List):
            self.depths = np.array(self.depths)

        self.gate_set = QUAGateSet(self.gate_set_choice, self.baseline_gate_name)
        self.n_qubits = len(self.qubits)
        self.dim = 2**self.n_qubits
        self.available_combinations = None
        self.coupling_map = None

    def as_dict(self):
        """
        Return the XEBConfig object as a dictionary
        """
        config_dict = {
            "seqs": self.seqs,
            "depths": self.depths.tolist() if isinstance(self.depths, np.ndarray) else self.depths,
            "n_shots": self.n_shots,
            "qubits": [qubit.name if isinstance(qubit, Transmon) else qubit for qubit in self.qubits],
            "baseline_gate_name": self.baseline_gate_name,
            "gate_set_choice": self.gate_set_choice,
            "two_qb_gate": self.two_qb_gate.name if self.two_qb_gate else None,
            "qubit_pairs": [pair.name if isinstance(pair, TransmonPair) else pair for pair in self.qubit_pairs],
            "coupling_map": list(self.coupling_map.get_edges()) if self.coupling_map else None,
            "available_combinations": self.available_combinations,
            "seed": self.seed,
        }
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict, machine: Optional[QuAM] = None):
        """
        Create an XEBConfig object from a dictionary that contains all relevant parameters.
        This method will usually be used to load a configuration from previously saved data from another run.

        Args:
            config_dict (Dict): Dictionary containing the configuration parameters
            machine (Optional[QuAM]): QuAM object containing the qubits and qubit pairs used in the experiment
        """
        qubits_names = config_dict["qubits"]
        qubits = [machine.qubits[qubit_name] if machine is not None else qubit_name for qubit_name in qubits_names]
        qubit_pairs_names = config_dict["qubit_pairs"]
        qubit_pairs = [
            machine.qubit_pairs[qubit_pair_name] if machine is not None else qubit_pair_name
            for qubit_pair_name in qubit_pairs_names
        ]
        two_qb_gate = QUAGate(config_dict["two_qb_gate"]) if config_dict["two_qb_gate"] else None
        if config_dict["gate_set_choice"] not in ["sw", "t"]:
            raise ValueError("Gate set choice must be either 'sw' or 't'")

        new_class = cls(
            seqs=config_dict["seqs"],
            depths=config_dict["depths"],
            n_shots=config_dict["n_shots"],
            qubits=qubits,
            baseline_gate_name=config_dict["baseline_gate_name"],
            gate_set_choice=config_dict["gate_set_choice"],
            two_qb_gate=two_qb_gate,
            qubit_pairs=qubit_pairs,
        )
        new_class.coupling_map = config_dict["coupling_map"]
        new_class.available_combinations = config_dict["available_combinations"]
        return new_class
