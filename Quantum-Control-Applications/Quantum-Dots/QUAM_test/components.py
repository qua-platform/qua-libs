from dataclasses import field
import numpy as np
from copy import copy
from typing import List, Union, Dict, Optional, ClassVar, Tuple
import warnings
from quam import QuamComponent
from quam.components.channels import QuamDict, IQChannel, SingleChannel, InOutSingleChannel, Channel, AmpValuesType, QuaNumberType, QuaExpressionType, StreamType, ChirpType, DigitalOutputChannel
from quam.components.pulses import Pulse
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass
from qm.qua import set_dc_offset, align, play, wait, amp, frame_rotation
from quam.utils import string_reference as str_ref
from qm.qua._dsl import (
    _PulseAmp,
    AmpValuesType,
    QuaNumberType,
    QuaExpressionType,
    ChirpType,
    StreamType,
)

# import macros

__all__ = ["StickyChannelAddon", "VirtualGateSet", "QuAM"]

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


# @quam_dataclass
# class Channel(QuamComponent):
#     """Base QuAM component for a channel, can be output, input or both.
#
#     Args:
#         operations (Dict[str, Pulse]): A dictionary of pulses to be played on this
#             channel. The key is the pulse label (e.g. "X90") and value is a Pulse.
#         id (str, int): The id of the channel, used to generate the name.
#             Can be a string, or an integer in which case it will add
#             `Channel._default_label`.
#     """
#
#     operations: Dict[str, Pulse] = field(default_factory=dict)
#
#     id: Union[str, int] = None
#     _default_label: ClassVar[str] = "ch"  # Used to determine name from id
#
#     digital_outputs: Dict[str, DigitalOutputChannel] = field(default_factory=dict)
#     # sticky: StickyChannelAddon = None
#
#     @property
#     def name(self) -> str:
#         cls_name = self.__class__.__name__
#
#         if self.id is not None:
#             if str_ref.is_reference(self.id):
#                 raise AttributeError(
#                     f"{cls_name}.name cannot be determined. "
#                     f"Please either set {cls_name}.id to a string or integer, "
#                     f"or {cls_name} should be an attribute of another QuAM component."
#                 )
#             if isinstance(self.id, str):
#                 return self.id
#             else:
#                 return f"{self._default_label}{self.id}"
#         if self.parent is None:
#             raise AttributeError(
#                 f"{cls_name}.name cannot be determined. "
#                 f"Please either set {cls_name}.id to a string or integer, "
#                 f"or {cls_name} should be an attribute of another QuAM component with "
#                 "a name."
#             )
#         if isinstance(self.parent, QuamDict):
#             return self.parent.get_attr_name(self)
#         if not hasattr(self.parent, "name"):
#             raise AttributeError(
#                 f"{cls_name}.name cannot be determined. "
#                 f"Please either set {cls_name}.id to a string or integer, "
#                 f"or {cls_name} should be an attribute of another QuAM component with "
#                 "a name."
#             )
#         return f"{self.parent.name}{str_ref.DELIMITER}{self.parent.get_attr_name(self)}"
#
#     @property
#     def pulse_mapping(self):
#         return {label: pulse.pulse_name for label, pulse in self.operations.items()}
#
#     def play(
#         self,
#         pulse_name: str,
#         amplitude_scale: Union[float, AmpValuesType] = None,
#         duration: QuaNumberType = None,
#         condition: QuaExpressionType = None,
#         chirp: ChirpType = None,
#         truncate: QuaNumberType = None,
#         timestamp_stream: StreamType = None,
#         continue_chirp: bool = False,
#         target: str = "",
#         validate: bool = True,
#     ):
#         """Play a pulse on this channel.
#
#         Args:
#             pulse_name (str): The name of the pulse to play. Should be registered in
#                 `self.operations`.
#             amplitude_scale (float, _PulseAmp): Amplitude scale of the pulse.
#                 Can be either a float, or qua.amp(float).
#             duration (int): Duration of the pulse in units of the clock cycle (4ns).
#                 If not provided, the default pulse duration will be used. It is possible
#                 to dynamically change the duration of both constant and arbitrary
#                 pulses. Arbitrary pulses can only be stretched, not compressed.
#             chirp (Union[(list[int], str), (int, str)]): Allows to perform
#                 piecewise linear sweep of the element's intermediate
#                 frequency in time. Input should be a tuple, with the 1st
#                 element being a list of rates and the second should be a
#                 string with the units. The units can be either: 'Hz/nsec',
#                 'mHz/nsec', 'uHz/nsec', 'pHz/nsec' or 'GHz/sec', 'MHz/sec',
#                 'KHz/sec', 'Hz/sec', 'mHz/sec'.
#             truncate (Union[int, QUA variable of type int]): Allows playing
#                 only part of the pulse, truncating the end. If provided,
#                 will play only up to the given time in units of the clock
#                 cycle (4ns).
#             condition (A logical expression to evaluate.): Will play analog
#                 pulse only if the condition's value is true. Any digital
#                 pulses associated with the operation will always play.
#             timestamp_stream (Union[str, _ResultSource]): (Supported from
#                 QOP 2.2) Adding a `timestamp_stream` argument will save the
#                 time at which the operation occurred to a stream. If the
#                 `timestamp_stream` is a string ``label``, then the timestamp
#                 handle can be retrieved with
#                 [`qm._results.JobResults.get`][qm.results.streaming_result_fetcher.StreamingResultFetcher] with the same
#                 ``label``.
#             validate (bool): If True (default), validate that the pulse is registered
#                 in Channel.operations
#
#         Note:
#             The `element` argument from `qm.qua.play()`is not needed, as it is
#             automatically set to `self.name`.
#
#         """
#         if validate and pulse_name not in self.operations:
#             raise KeyError(
#                 f"Operation '{pulse_name}' not found in channel '{self.name}'"
#             )
#
#         if amplitude_scale is not None:
#             if not isinstance(amplitude_scale, _PulseAmp):
#                 amplitude_scale = amp(amplitude_scale)
#             pulse = pulse_name * amplitude_scale
#         else:
#             pulse = pulse_name
#
#         # At the moment, self.name is not defined for Channel because it could
#         # be a property or dataclass field in a subclass.
#         # # TODO Find elegant solution for Channel.name.
#         play(
#             pulse=pulse,
#             element=self.name,
#             duration=duration,
#             condition=condition,
#             chirp=chirp,
#             truncate=truncate,
#             timestamp_stream=timestamp_stream,
#             continue_chirp=continue_chirp,
#             target=target,
#         )
#
#     def wait(self, duration: QuaNumberType, *other_elements: Union[str, "Channel"]):
#         """Wait for the given duration on all provided elements without outputting anything.
#
#         Duration is in units of the clock cycle (4ns)
#
#         Args:
#             duration (Union[int,QUA variable of type int]): time to wait in
#                 units of the clock cycle (4ns). Range: [4, $2^{31}-1$]
#                 in steps of 1.
#             *other_elements (Union[str,sequence of str]): elements to wait on,
#                 in addition to this channel
#
#         Warning:
#             In case the value of this is outside the range above, unexpected results may occur.
#
#         Note:
#             The current channel element is always included in the wait operation.
#
#         Note:
#             The purpose of the `wait` operation is to add latency. In most cases, the
#             latency added will be exactly the same as that specified by the QUA variable or
#             the literal used. However, in some cases an additional computational latency may
#             be added. If the actual wait time has significance, such as in characterization
#             experiments, the actual wait time should always be verified with a simulator.
#         """
#         other_elements_str = [
#             element if isinstance(element, str) else str(element)
#             for element in other_elements
#         ]
#         wait(duration, self.name, *other_elements_str)
#
#     def align(self, *other_elements):
#         if not other_elements:
#             align()
#         else:
#             other_elements_str = [
#                 element if isinstance(element, str) else str(element)
#                 for element in other_elements
#             ]
#             align(self.name, *other_elements_str)
#
#     def frame_rotation(self, angle: QuaNumberType):
#         r"""Shift the phase of the channel element's oscillator by the given angle.
#
#         This is typically used for virtual z-rotations.
#
#         Note:
#             The fixed point format of QUA variables of type fixed is 4.28, meaning the
#             phase must be between $-8$ and $8-2^{28}$. Otherwise the phase value will be
#             invalid. It is therefore better to use `frame_rotation_2pi()` which avoids
#             this issue.
#
#         Note:
#             The phase is accumulated with a resolution of 16 bit.
#             Therefore, *N* changes to the phase can result in a phase (and amplitude)
#             inaccuracy of about :math:`N \cdot 2^{-16}`. To null out this accumulated
#             error, it is recommended to use `reset_frame(el)` from time to time.
#
#         Args:
#             angle (Union[float, QUA variable of type fixed]): The angle to
#                 add to the current phase (in radians)
#             *elements (str): a single element whose oscillator's phase will
#                 be shifted. multiple elements can be given, in which case
#                 all of their oscillators' phases will be shifted
#
#         """
#         frame_rotation(angle, self.name)
#
#     def _config_add_controller(
#         self, config: Dict[str, dict], controller_name: str
#     ) -> Dict[str, dict]:
#         """Adds a controller to the config if it doesn't exist, and returns its config.
#
#         config.controllers.<controller_name> will be created if it doesn't exist.
#         It will also add the analog_outputs, digital_outputs, and analog_inputs keys
#
#         Args:
#             config (dict): The QUA config that's in the process of being generated.
#             controller_name (str): The name of the controller.
#
#         Returns:
#             Dict[str, dict]: The config entry for the controller.
#         """
#         config["controllers"].setdefault(controller_name, {})
#         controller_cfg = config["controllers"][controller_name]
#         for key in ["analog_outputs", "digital_outputs", "analog_inputs"]:
#             controller_cfg.setdefault(key, {})
#
#         return controller_cfg
#
#     def _config_add_digital_outputs(self, config: Dict[str, dict]) -> None:
#         """Adds the digital outputs to the QUA config.
#
#         config.elements.<element_name>.digitalInputs will be updated with the digital
#         outputs of this channel.
#
#         Note that the digital outputs are added separately to the controller config in
#         `DigitalOutputChannel.apply_to_config`.
#
#         Args:
#             config (dict): The QUA config that's in the process of being generated.
#         """
#         if not self.digital_outputs:
#             return
#
#         element_cfg = config["elements"][self.name]
#         element_cfg.setdefault("digitalInputs", {})
#
#         for name, digital_output in self.digital_outputs.items():
#             digital_cfg = digital_output.generate_element_config()
#             element_cfg["digitalInputs"][name] = digital_cfg
#
#     def apply_to_config(self, config: Dict[str, dict]) -> None:
#         """Adds this Channel to the QUA configuration.
#
#         config.elements.<element_name> will be created, and the operations are added.
#
#         Args:
#             config (dict): The QUA config that's in the process of being generated.
#
#         Raises:
#             ValueError: If the channel already exists in the config.
#         """
#         if self.name in config["elements"]:
#             raise ValueError(
#                 f"Cannot add channel {self.name} to the config because it already "
#                 f"exists. Existing entry: {config['elements'][self.name]}"
#             )
#         config["elements"][self.name] = {"operations": self.pulse_mapping}
#
#         self._config_add_digital_outputs(config)
# @quam_dataclass
# class SingleChannel(Channel):
#     """QuAM component for a single (not IQ) output channel.
#
#     Args:
#         operations (Dict[str, Pulse]): A dictionary of pulses to be played on this
#             channel. The key is the pulse label (e.g. "X90") and value is a Pulse.
#         id (str, int): The id of the channel, used to generate the name.
#             Can be a string, or an integer in which case it will add
#             `Channel._default_label`.
#         opx_output (Tuple[str, int]): Channel output port from the OPX perspective,
#             a tuple of (controller_name, port).
#         filter_fir_taps (List[float]): FIR filter taps for the output port.
#         filter_iir_taps (List[float]): IIR filter taps for the output port.
#         opx_output_offset (float): DC offset for the output port.
#         intermediate_frequency (float): Intermediate frequency of OPX output, default
#             is None.
#         sticky (Sticky): Optional sticky parameters for the channel, i.e. defining
#             whether successive pulses are applied w.r.t the previous pulse or w.r.t 0 V.
#             If not specified, this channel is not sticky.
#     """
#
#     opx_output: Tuple[str, int]
#     filter_fir_taps: List[float] = None
#     filter_iir_taps: List[float] = None
#
#     opx_output_offset: float = 0.0
#     intermediate_frequency: float = None
#     #
#     # sticky = StickyChannelAddon = None
#
#     def apply_to_config(self, config: dict):
#         """Adds this SingleChannel to the QUA configuration.
#
#         See [`QuamComponent.apply_to_config`][quam.core.quam_classes.QuamComponent.apply_to_config]
#         for details.
#         """
#         # Add pulses & waveforms
#         super().apply_to_config(config)
#
#         if str_ref.is_reference(self.name):
#             raise AttributeError(
#                 f"Channel {self.get_reference()} cannot be added to the config because"
#                 " it doesn't have a name. Either set channel.id to a string or"
#                 " integer, or channel should be an attribute of another QuAM component"
#                 " with a name."
#             )
#
#         element_config = config["elements"][self.name]
#         element_config["singleInput"] = {"port": tuple(self.opx_output)}
#
#         if self.intermediate_frequency is not None:
#             element_config["intermediate_frequency"] = self.intermediate_frequency
#
#         controller_name, port = self.opx_output
#         controller_cfg = self._config_add_controller(config, controller_name)
#         analog_output = controller_cfg["analog_outputs"].setdefault(port, {})
#         # If no offset specified, it will be added at the end of the config generation
#         offset = self.opx_output_offset
#         if offset is not None:
#             if abs(analog_output.get("offset", offset) - offset) > 1e-4:
#                 warnings.warn(
#                     f"Channel {self.name} has conflicting output offsets: "
#                     f"{analog_output['offset']} V and {offset} V. Multiple channel "
#                     f"elements are trying to set different offsets to port {port}. "
#                     f"Using the last offset {offset} V"
#                 )
#             analog_output["offset"] = offset
#
#         if self.filter_fir_taps is not None:
#             output_filter = analog_output.setdefault("filter", {})
#             output_filter["feedforward"] = list(self.filter_fir_taps)
#
#         if self.filter_iir_taps is not None:
#             output_filter = analog_output.setdefault("filter", {})
#             output_filter["feedback"] = list(self.filter_iir_taps)
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


# @quam_dataclass
# class ReadoutResonator(InOutIQChannel):
#     """QuAM component for a readout resonator
#
#     :params depletion_time: the resonator depletion time in ns.
#     :params frequency_bare: the bare resonator frequency in Hz.
#     """
#
#     depletion_time: int = 1000
#     frequency_bare: float = 0.0
#
#     @property
#     def f_01(self):
#         """The optimal frequency for discriminating the qubit between |0> and |1> in Hz"""
#         return self.frequency_converter_up.LO_frequency + self.intermediate_frequency


# @quam_dataclass
# class Transmon(QuamComponent):
#     """
#     Example QuAM component for a transmon qubit.
#
#     Args:
#         thermalization_time (int): An integer.
#         T1 (str): A string.
#     """
#
#     id: Union[int, str]
#
#     xy: IQChannel = None
#     z: FluxLine = None
#
#     resonator: ReadoutResonator = None
#
#     T1: int = 10_000
#     T2ramsey: int = 10_000
#     T2echo: int = 10_000
#     thermalization_time_factor: int = 5
#     anharmonicity: int = 150e6
#
#     @property
#     def thermalization_time(self):
#         return self.thermalization_time_factor * self.T1
#
#     @property
#     def f_01(self):
#         """The 0-1 (g-e) transition frequency in Hz"""
#         return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency
#
#     @property
#     def f_12(self):
#         """The 0-2 (e-f) transition frequency in Hz"""
#         return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency - self.anharmonicity
#
#     @property
#     def name(self):
#         return self.id if isinstance(self.id, str) else f"q{self.id}"


# @quam_dataclass
# class QuAM(QuamRoot):
#     """Example QuAM root component."""
#
#     @classmethod
#     def load(self, *args, **kwargs) -> "QuAM":
#         return super().load(*args, **kwargs)
#
#     octave: Octave = None
#
#     qubits: Dict[str, Transmon] = field(default_factory=dict)
#     wiring: dict = field(default_factory=dict)
#     network: dict = field(default_factory=dict)
#
#     active_qubit_names: List[str] = field(default_factory=list)
#
#     # @property
#     # def network(self) -> Dict[str, str]:
#     #     return {"host": "172.16.33.101", "cluster_name": "Cluster_81"}
#
#     @property
#     def active_qubits(self) -> List[Transmon]:
#         """Return the list of active qubits"""
#         return [self.qubits[q] for q in self.active_qubit_names]
#
#     @property
#     def get_depletion_time(self) -> int:
#         """Return the longest depletion time amongst the active qubits"""
#         return max([q.resonator.depletion_time for q in self.active_qubits])
#
#     @property
#     def get_thermalization_time(self) -> int:
#         """Return the longest thermalization time amongst the active qubits"""
#         return max([q.thermalization_time for q in self.active_qubits])
#
#     def apply_all_flux_to_min(self) -> None:
#         """Apply the offsets that bring all the active qubits to the minimum frequency point."""
#         align()
#         for q in self.active_qubits:
#             q.z.to_min()
#         align()
#
#     # def connect(self):
#     #     from qm import QuantumMachinesManager
#     #     return QuantumMachinesManager(
#     #         host=self.network["host"], cluster_name=self.network["cluster_name"], octave=octave_config
#     #     )

@quam_dataclass
class QuAM(QuamRoot):
    gates: Dict[str, SingleChannel] = field(default_factory=dict)
    resonator: InOutSingleChannel = None
    virtual_gate_set: VirtualGateSet = None
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)
    # virtual_gate_set_twin: VirtualGateSet = None

    def align_gates(self, ):
        align(*self.gates)

    def align_all(self,):
        align(self.resonator.name, *self.gates)