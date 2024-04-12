from dataclasses import field
import numpy as np
from copy import copy
from typing import List, Union, Dict, Optional, ClassVar, Tuple
import warnings
from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutSingleChannel, Channel, DigitalOutputChannel
from quam.components.channels import QuamDict, AmpValuesType, QuaNumberType, QuaExpressionType, StreamType, ChirpType
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

__all__ = ["QuAM"]

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
class QuAM(QuamRoot):
    test: SingleChannel = None
    qdac_trigger: Dict[str, Channel] = field(default_factory=dict)
    # Architecture and network settings
    wiring: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)

    def align_gates(
        self,
    ):
        align(*self.gates)

    def align_all(
        self,
    ):
        align(self.resonator.name, *self.gates)

    # @property
    # def active_qubits(self) -> List[Transmon]:
    #     """Return the list of active qubits"""
    #     return [self.qubits[q] for q in self.active_qubit_names]

    def connect(self):
        from qm import QuantumMachinesManager

        return QuantumMachinesManager(
            host=self.network["host"], cluster_name=self.network["cluster_name"], octave=self.octave.get_octave_config()
        )
