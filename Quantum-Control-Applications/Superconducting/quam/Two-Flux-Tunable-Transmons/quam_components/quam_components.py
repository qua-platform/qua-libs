from quam.core import QuamRoot, quam_dataclass
from quam.components.octave import Octave
from .Transmon_component import Transmon

from qm.qua import align
from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.results.data_handler import DataHandler

from dataclasses import field
from typing import List, Dict, ClassVar


__all__ = ["QuAM"]


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)

    octave: Octave = None

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)

    _data_handler: ClassVar[DataHandler] = None

    @property
    def data_handler(self) -> DataHandler:
        """Return the existing data handler or open a new one to conveniently handle data saving."""
        if self._data_handler is None:
            self._data_handler = DataHandler(root_data_folder=self.network["data_folder"])
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @property
    def active_qubits(self) -> List[Transmon]:
        """Return the list of active qubits."""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def get_depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits."""
        return max([q.resonator.depletion_time for q in self.active_qubits])

    @property
    def get_thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits."""
        return max([q.thermalization_time for q in self.active_qubits])

    def apply_all_flux_to_min(self) -> None:
        """Apply the offsets that bring all the active qubits to the minimum frequency point."""
        align()
        for q in self.active_qubits:
            q.z.to_min()
        align()

    def connect(self) -> QuantumMachinesManager:
        """Open a Quantum Machine Manager with the credentials ("host" and "cluster_name") as defined in the network file.

        Returns: the opened Quantum Machine Manager.
        """
        return QuantumMachinesManager(
            host=self.network["host"], cluster_name=self.network["cluster_name"], octave=self.octave.get_octave_config()
        )

    def calibrate_active_qubits(self, QM: QuantumMachine) -> None:
        """Calibrate the Octave ports for all the active qubits.

        Args:
            QM (QuantumMachine): the running quantum machine.
        """
        for name in self.active_qubit_names:
            self.qubits[name].calibrate_octave(QM)
