import os
from pathlib import Path
from qm.octave import QmOctaveConfig
from quam.components import FrequencyConverter
from quam.core import QuamRoot, quam_dataclass
from quam.components.octave import Octave
from quam.components.ports import FEMPortsContainer, OPXPlusPortsContainer
from quam_builder.components.superconducting.qubit_pair.flux_tunable_transmons import TransmonPair
from quam_builder.components.superconducting.qubit import AnyTransmon
from quam_builder.components.superconducting.architectural_elements.readout_resonator import ReadoutResonatorMW, ReadoutResonatorIQ
from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.results.data_handler import DataHandler

from dataclasses import field
from typing import List, Dict, ClassVar, Optional, Sequence, Union

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

__all__ = ["BaseQuAM"]

# from ....experiments.node_parameters import QubitsExperimentNodeParameters
# from quam_builder.batchable_list import make_batchable_list

@quam_dataclass
class BaseQuAM(QuamRoot):
    """Example QuAM root component."""

    octaves: Dict[str, Octave] = field(default_factory=dict)
    mixers: Dict[str, FrequencyConverter] = field(default_factory=dict)

    qubits: Dict[str, AnyTransmon] = field(default_factory=dict)
    qubit_pairs: Dict[str, TransmonPair] = field(default_factory=dict)
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)
    active_qubit_pair_names: List[str] = field(default_factory=list)

    ports: Union[FEMPortsContainer, OPXPlusPortsContainer] = None

    _data_handler: ClassVar[DataHandler] = None
    qmm: ClassVar[Optional[QuantumMachinesManager]] = None

    @classmethod
    def get_quam_state_path(cls) -> Optional[Path]:
        qualibrate_config_path = Path.home() / ".qualibrate" / "config.toml"

        if "QUAM_STATE_PATH" in os.environ:
            return Path(os.environ["QUAM_STATE_PATH"])
        elif qualibrate_config_path.exists():
            config = tomllib.loads(qualibrate_config_path.read_text())
            quam_state_path = config.get("quam", {}).get("state_path", None)
            return Path(quam_state_path)
        else:
            return None

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        if not args:
            quam_state_path = cls.get_quam_state_path()
            if quam_state_path is None:
                raise ValueError(
                    "No path argument provided to load the QuAM state. "
                    "Please provide a path or set the 'QUAM_STATE_PATH' environment variable. "
                    "See the README for instructions."
                )

            args = (quam_state_path,)

        return super().load(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        if not args:
            if "QUAM_STATE_PATH" in os.environ:
                args = (os.environ["QUAM_STATE_PATH"],)
            else:
                raise ValueError(
                    "No path argument provided to load the QuAM state. "
                    "Please provide a path or set the 'QUAM_STATE_PATH' environment variable. "
                    "See the README for instructions."
                )
        return super().load(*args, **kwargs)

    def save(
            self,
            path: Union[Path, str] = None,
            content_mapping: Dict[Union[Path, str], Sequence[str]] = None,
            include_defaults: bool = False,
            ignore: Sequence[str] = None,
    ):
        if path is None and "QUAM_STATE_PATH" in os.environ:
            path = os.environ["QUAM_STATE_PATH"]

        super().save(path, content_mapping, include_defaults, ignore)

    def get_octave_config(self) -> QmOctaveConfig:
        """Return the Octave configuration."""
        octave_config = None
        for octave in self.octaves.values():
            if octave_config is None:
                octave_config = octave.get_octave_config()
        return octave_config

    def connect(self) -> QuantumMachinesManager:
        """Open a Quantum Machine Manager with the credentials ("host" and "cluster_name") as defined in the network file.

        Returns: the opened Quantum Machine Manager.
        """
        settings = dict(
            host=self.network["host"],
            cluster_name=self.network["cluster_name"],
            octave=self.get_octave_config(),
        )
        if "port" in self.network:
            settings["port"] = self.network["port"]
        self.qmm = QuantumMachinesManager(**settings) # TODO: how to fix this warning?
        return self.qmm

    def calibrate_octave_ports(self, QM: QuantumMachine) -> None:
        """Calibrate the Octave ports for all the active qubits.

        Args:
            QM (QuantumMachine): the running quantum machine.
        """
        from qm.octave.octave_mixer_calibration import NoCalibrationElements

        for name in self.active_qubit_names:
            try:
                self.qubits[name].calibrate_octave(QM)
            except NoCalibrationElements:
                print(f"No calibration elements found for {name}. Skipping calibration.")

    def get_qubits_used_in_node(self, node_parameters) -> Sequence[AnyTransmon]:
        if node_parameters.qubits is None or node_parameters.qubits == "":
            qubits = self.active_qubits
        else:
            qubits = [self.qubits[q] for q in node_parameters.qubits]
        return qubits
        # return make_batchable_list(qubits, node_parameters)

    def get_resonators_used_in_node(self, node_parameters) -> Sequence[
        Union[ReadoutResonatorIQ, ReadoutResonatorMW]]:
        resonators = [qubit.resonator for qubit in self.get_qubits_used_in_node(node_parameters)]
        return resonators
        # return make_batchable_list(resonators, node_parameters)

    @property
    def data_handler(self) -> DataHandler:
        """Return the existing data handler or open a new one to conveniently handle data saving."""
        if self._data_handler is None:
            self._data_handler = DataHandler(root_data_folder=self.network["data_folder"]) # TODO: how to fix this warning?
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @property
    def active_qubits(self) -> List[AnyTransmon]:
        """Return the list of active qubits."""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def active_qubit_pairs(self) -> List[TransmonPair]:
        """Return the list of active qubits."""
        return [self.qubit_pairs[q] for q in self.active_qubit_pair_names]

    @property
    def depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits."""
        return max(q.resonator.depletion_time for q in self.active_qubits)

    @property
    def thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits."""
        return max(q.thermalization_time for q in self.active_qubits)
