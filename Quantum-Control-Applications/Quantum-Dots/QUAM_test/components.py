from dataclasses import field
import numpy as np
from copy import copy
from typing import List, Union, Dict, Optional
from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel, Channel, AmpValuesType, QuaNumberType
from quam.components.pulses import Pulse
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass
from qm.qua import set_dc_offset, align
# import macros

__all__ = ["StickyChannelAddon", "VirtualGateSet", "ReadoutResonator", "QuAM"]

################################### ADDONS #############################################################################
########################################################################################################################
@quam_dataclass
class StickyChannelAddon(QuamComponent):
    duration: int
    enabled: bool = True
    analog: bool = True
    digital: bool = True

    @property
    def channel(self) -> Optional["Channel"]:
        """If the parent is a channel, returns the parent, otherwise returns None."""
        if isinstance(self.parent, Channel):
            return self.parent
        else:
            return

    @property
    def config_settings(self):
        if self.channel is not None:
            return {"after": [self.channel]}

    def apply_to_config(self, config: dict) -> None:
        if self.channel is None:
            return

        if not self.enabled:
            return

        config["elements"][self.channel.name]["sticky"] = {
            "analog": self.analog,
            "digital": self.digital,
            "duration": self.duration,
        }

@quam_dataclass
class VirtualPulse(Pulse):
    amplitudes: Dict[str, float]
    # pulses: List[Pulse] = None  # Should be added later

    @property
    def virtual_gate_set(self):
        virtual_gate_set = self.parent.parent
        assert isinstance(virtual_gate_set, VirtualGateSet)
        return virtual_gate_set

    def waveform_function(self): ...


########################################################################################################################

@quam_dataclass
class VirtualGateSet(QuamComponent):
    gates: List[SingleChannel]
    virtual_gates: Dict[str, List[float]]

    pulse_defaults: List[Pulse] = field(default_factory=list)
    operations: Dict[str, VirtualPulse] = field(default_factory=dict)

    @property
    def config_settings(self):
        return {"after": self.gates}

    def convert_amplitudes(self, **virtual_gate_amplitudes):
        gate_amplitudes = np.zeros(len(self.gates))
        for virtual_gate_name, amplitude in virtual_gate_amplitudes.items():
            scales = self.virtual_gates[virtual_gate_name]
            gate_amplitudes += amplitude * np.array(scales)

        return gate_amplitudes

    def play(
        self,
        pulse_name,
        amplitude_scale: Union[float, AmpValuesType] = None,
        duration: QuaNumberType = None,
        **kwargs,
    ):
        """Play a pulse on all gates in the virtual gate set

        Args:
            pulse_name: The name of the pulse to play
            amplitude_scale: The amplitude scale to apply to the pulse
            duration: The duration of the pulse
            **kwargs: Additional kwargs to pass to the play function
        """
        for gate in self.gates:
            gate.play(
                pulse_name,
                validate=False,
                amplitude_scale=amplitude_scale,
                duration=duration,
                **kwargs,
            )

    def apply_to_config(self, config: dict) -> None:
        for operation_name, operation in self.operations.items():
            gate_pulses = [copy(pulse) for pulse in self.pulse_defaults]
            gate_amplitudes = self.convert_amplitudes(**operation.amplitudes)

            for gate, pulse, amplitude in zip(self.gates, gate_pulses, gate_amplitudes):
                pulse.id = operation_name
                pulse.amplitude = amplitude
                pulse.length = operation.length
                pulse.parent = None  # Reset parent so it can be attached to new parent
                pulse.parent = gate
                pulse.apply_to_config(config)

                element_config = config["elements"][gate.name]
                element_config["operations"][operation_name] = pulse.pulse_name


@quam_dataclass
class FluxLine(SingleChannel):
    """Example QuAM component for a transmon qubit."""

    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0

    def to_independent_idle(self):  # TODO: put the functions here
        set_dc_offset(self.name, "single", self.independent_offset)

    def to_joint_idle(self):
        set_dc_offset(self.name, "single", self.joint_offset)

    def to_min(self):
        set_dc_offset(self.name, "single", self.min_offset)


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

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    @classmethod
    def load(self, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)

    octave: Octave = None

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)

    # @property
    # def network(self) -> Dict[str, str]:
    #     return {"host": "172.16.33.101", "cluster_name": "Cluster_81"}

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

    # def connect(self):
    #     from qm import QuantumMachinesManager
    #     return QuantumMachinesManager(
    #         host=self.network["host"], cluster_name=self.network["cluster_name"], octave=octave_config
    #     )

@quam_dataclass
class QuAM(QuamRoot):
    gates: Dict[str, SingleChannel] = field(default_factory=dict)
    resonator: InOutSingleChannel_M = None
    virtual_gate_set: VirtualGateSetWithRamps = None
    virtual_gate_set_twin: VirtualGateSetWithRamps = None

    def align_gates(self, ):
        align(*self.gates)

    def align_all(self,):
        align(self.resonator.name, *self.gates)