from quam.core import QuamRoot, quam_dataclass
from quam.components.octave import Octave
from .transmon import Transmon
from .transmon_pair import TransmonPair

from qm.qua import align
from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.results.data_handler import DataHandler

from dataclasses import field
from typing import List, Dict, ClassVar, Any

__all__ = ["QuAM"]


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    octaves: Dict[str, Octave] = field(default_factory=dict)

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    qubit_pairs: List[TransmonPair] = field(default_factory=list)
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)

    _data_handler: ClassVar[DataHandler] = None

    @classmethod
    def load(cls, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)

    @property
    def data_handler(self) -> DataHandler:
        """Return the existing data handler or open a new one to conveniently handle data saving."""
        if self._data_handler is None:
            self._data_handler = DataHandler(
                root_data_folder=self.network["data_folder"]
            )
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @property
    def active_qubits(self) -> List[Transmon]:
        """Return the list of active qubits."""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits."""
        return max(q.resonator.depletion_time for q in self.active_qubits)

    @property
    def thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits."""
        return max(q.thermalization_time for q in self.active_qubits)

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
        settings = dict(
            host=self.network["host"],
            cluster_name=self.network["cluster_name"],
            octave=self.get_octave_config(),
        )
        if "port" in self.network:
            settings["port"] = self.network["port"]
        return QuantumMachinesManager(**settings)

    def get_octave_config(self) -> dict:
        """Return the Octave configuration."""
        octave_config = None
        for octave in self.octaves.values():
            if octave_config is None:
                octave_config = octave.get_octave_config()
            else:
                octave_config.add_device_info(octave.name, octave.ip, octave.port)

        return octave_config

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
                print(
                    f"No calibration elements found for {name}. Skipping calibration."
                )

    def generate_config(self) -> Dict[str, Any]:
        config = super().generate_config()

        # make sure to not override existing fem
        lf_fem_remapping = {1: 3, 2: 4}
        mw_fem_dummies = [1]

        fems = config["controllers"]["con1"]["fems"]

        for key in lf_fem_remapping:
            fems[lf_fem_remapping[key]] = fems[key]
            del fems[key]

        for mw_fem_dummy in mw_fem_dummies:
            fems[mw_fem_dummy] = {
                "type": "MW",
                "analog_outputs": {}
            }

        octave_rf_outputs = config["octaves"]["octave1"].get("RF_outputs", {})
        octave_rf_inputs = config["octaves"]["octave1"].get("RF_inputs", {})
        for octave_rf_config in [octave_rf_inputs, octave_rf_outputs]:
            for octave_port in octave_rf_config.values():
                for analog_port in ["I_connection", "Q_connection"]:
                    if analog_port in octave_port:
                        original_port = octave_port[analog_port]
                        if original_port[1] in lf_fem_remapping:
                            remapped_port = (original_port[0], lf_fem_remapping[original_port[1]], original_port[2])
                            octave_port[analog_port] = remapped_port

        elements = config["elements"]
        for element in elements.values():
            if "digitalInputs" in element:
                for operation in element["digitalInputs"]:
                    original_port = element["digitalInputs"][operation]["port"]
                    element["digitalInputs"][operation]["port"] = (original_port[0], lf_fem_remapping[original_port[1]], original_port[2])

            if "singleInput" in element:
                original_port = element["singleInput"]["port"]
                element["singleInput"]["port"] = (original_port[0], lf_fem_remapping[original_port[1]], original_port[2])

        return config
