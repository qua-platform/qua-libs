from dataclasses import field
from typing import List, Union, Dict
from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass

from qm.qua import align
from qualang_tools.results.data_handler import DataHandler
from typing import ClassVar
from qualang_tools.octave_tools import octave_calibration_tool
from qm import QuantumMachine


__all__ = ["Transmon", "FluxLine", "ReadoutResonator", "QuAM"]


@quam_dataclass
class FluxLine(SingleChannel):
    """Example QuAM component for a transmon qubit."""

    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0

    def to_independent_idle(self):
        self.set_dc_offset(self.independent_offset)

    def to_joint_idle(self):
        self.set_dc_offset(self.joint_offset)

    def to_min(self):
        self.set_dc_offset(self.min_offset)


@quam_dataclass
class ReadoutResonator(InOutIQChannel):
    """QuAM component for a readout resonator

    :params depletion_time: the resonator depletion time in ns.
    :params frequency_bare: the bare resonator frequency in Hz.
    """

    depletion_time: int = 1000
    frequency_bare: float = 0.0

    @property
    def f_01(self):
        """The optimal frequency for discriminating the qubit between |0> and |1> in Hz"""
        return self.frequency_converter_up.LO_frequency + self.intermediate_frequency


@quam_dataclass
class Transmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        thermalization_time (int): An integer.
        T1 (str): A string.
    """

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None

    resonator: ReadoutResonator = None

    T1: int = 10_000
    T2ramsey: int = 10_000
    T2echo: int = 10_000
    thermalization_time_factor: int = 5
    anharmonicity: int = 150e6

    @property
    def thermalization_time(self):
        return self.thermalization_time_factor * self.T1

    @property
    def f_01(self):
        """The 0-1 (g-e) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency

    @property
    def f_12(self):
        """The 0-2 (e-f) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency - self.anharmonicity

    def calibrate_octave(self, QM: QuantumMachine):
        octave_calibration_tool(
            QM,
            self.xy.name,
            lo_frequencies=self.xy.frequency_converter_up.LO_frequency,
            intermediate_frequencies=self.xy.intermediate_frequency,
        )
        octave_calibration_tool(
            QM,
            self.resonator.name,
            lo_frequencies=self.resonator.frequency_converter_up.LO_frequency,
            intermediate_frequencies=self.resonator.intermediate_frequency,
        )

    def set_gate_shape(self, gate_shape: str):
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            self.xy.operations[gate] = "#./" + gate + "_" + gate_shape

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


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
    def data_handler(self):
        if self._data_handler is None:
            self._data_handler = DataHandler(root_data_folder=self.network["data_folder"])
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @property
    def active_qubits(self) -> List[Transmon]:
        """Return the list of active qubits"""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def get_depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits"""
        return max([q.resonator.depletion_time for q in self.active_qubits])

    @property
    def get_thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits"""
        return max([q.thermalization_time for q in self.active_qubits])

    def apply_all_flux_to_min(self) -> None:
        """Apply the offsets that bring all the active qubits to the minimum frequency point."""
        align()
        for q in self.active_qubits:
            q.z.to_min()
        align()

    @property
    def get_data_handler(self) -> DataHandler:
        return DataHandler(root_data_folder=self.network["data_folder"])

    def connect(self):
        from qm import QuantumMachinesManager

        return QuantumMachinesManager(
            host=self.network["host"], cluster_name=self.network["cluster_name"], octave=self.octave.get_octave_config()
        )

    def calibrate_active_qubits(self, QM: QuantumMachine):
        for name in self.active_qubit_names:
            self.qubits[name].calibrate_octave(QM)
