from quam.core import quam_dataclass
from quam.components.channels import IQChannel, Pulse
from quam import QuamComponent
from .FluxLine_component import FluxLine
from .ReadoutResonator_component import ReadoutResonator
from qualang_tools.octave_tools import octave_calibration_tool
from qm import QuantumMachine
from typing import Union


__all__ = ["Transmon"]


@quam_dataclass
class Transmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        xy (IQChannel): The xy drive component.
        z (FluxLine): The z drive component.
        resonator (ReadoutResonator): The readout resonator component.
        T1 (int): The transmon T1 in ns.
        T2ramsey (int): The transmon T2* in ns.
        T2echo (int): The transmon T2 in ns.
        thermalization_time_factor (int): thermalization time in units of T1.
        anharmonicity (int, float): the transmon anharmonicity in Hz.
    """
    # TODO: update with inferred frequencies

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None
    resonator: ReadoutResonator = None

    T1: int = 10_000
    T2ramsey: int = 10_000
    T2echo: int = 10_000
    thermalization_time_factor: int = 5
    anharmonicity: int = 150e6
    sigma_time_factor: int = 5

    @property
    def sigma(self, operation: Pulse):
        return operation.length / self.sigma_time_factor

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns."""
        return self.thermalization_time_factor * self.T1

    @property
    def f_01(self):
        """The 0-1 (g-e) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency

    @property
    def f_12(self):
        """The 0-2 (e-f) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency - self.anharmonicity

    def calibrate_octave(self, QM: QuantumMachine) -> None:
        """Calibrate the Octave channels (xy and resonator) linked to this transmon for the LO frequency, intermediate
        frequency and Octave gain as defined in the state.

        Args:
            QM (QuantumMachine): the running quantum machine.
        """
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

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            self.xy.operations[gate] = f"#./{gate}_{gate_shape}"

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
