from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import field
from copy import copy

from qm.qua import align, declare, fixed, frame_rotation_2pi

from quam.components.pulses import Pulse
from quam.core import quam_dataclass, QuamComponent
from quam.utils import string_reference as str_ref


__all__ = ["TwoQubitGate", "CZGate", "CZWithCompensationGate", "CNotGate_TC", "CNotGate_CT"]


@quam_dataclass
class TwoQubitGate(QuamComponent, ABC):
    
    extras: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def transmon_pair(self):
        from ..transmon_pair import TransmonPair

        if isinstance(self.parent, TransmonPair):
            return self.parent
        elif hasattr(self.parent, "parent") and isinstance(self.parent.parent, TransmonPair):
            return self.parent.parent
        else:
            raise AttributeError("TwoQubitGate is not attached to a QubitPair. 2Q_gate: {self}")

    @property
    def qubit_control(self):
        return self.transmon_pair.qubit_control

    @property
    def qubit_target(self):
        return self.transmon_pair.qubit_target
    
    @property
    def coupler(self):
        return self.transmon_pair.coupler    

    def __call__(self):
        self.execute()


@quam_dataclass
class CZGate(TwoQubitGate):
    """CZ Operation for a qubit pair"""

    # Pulses will be added to qubit elements
    # The reason we don't add "flux_to_q1" directly to q1.z is because it is part of
    # the CZ operation, i.e. it is only applied as part of a CZ operation

    flux_pulse_control: Pulse
    
    pre_wait: int = 4

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0
    

    @property
    def gate_label(self) -> str:
        try:
            return self.parent.get_attr_name(self)
        except AttributeError:
            return "CZ"

    @property
    def flux_pulse_control_label(self) -> str:
        if self.flux_pulse_control.id is not None:
            pulse_label = self.flux_pulse_control.id
        else:
            pulse_label = "flux_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    def execute(self, amplitude_scale=None):        
        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )
        
        self.transmon_pair.align()
        frame_rotation_2pi(self.phase_shift_control, self.qubit_control.xy.name)
        frame_rotation_2pi(self.phase_shift_target, self.qubit_target.xy.name)
        self.qubit_control.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.qubit_target.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.transmon_pair.align()

    @property
    def config_settings(self):
        return {"after": [self.qubit_control.z]}

    def apply_to_config(self, config: dict) -> None:
        pulse = copy(self.flux_pulse_control)
        pulse.id = self.flux_pulse_control_label
        pulse.parent = None  # Reset parent so it can be attached to new parent
        pulse.parent = self.qubit_control.z

        if self.flux_pulse_control_label in self.qubit_control.z.operations:
            raise ValueError(
                "Pulse name already exists in pulse operations. "
                f"Channel: {self.qubit_control.z.get_reference()}, "
                f"Pulse: {self.flux_pulse_control.get_reference()}, "
                f"Pulse name: {self.flux_pulse_control_label}"
            )

        pulse.apply_to_config(config)

        element_config = config["elements"][self.qubit_control.z.name]
        element_config["operations"][self.flux_pulse_control_label] = pulse.pulse_name


@quam_dataclass
class CZWithCompensationGate(CZGate):
    compensations : list   # Extra qubits that are not part of the gate but are shifted when the gate to be executed. Expected format: [{"qubit": qubit_object,"shift" : float, "phase" : float}]

    def execute(self, *args, **kwargs):
        if not self.compensations:
            super().execute(*args, **kwargs)
            return
        
        compensation_qubits = [compensation["qubit"] for compensation in self.compensations]
        qubits = [self.qubit_control, self.qubit_target, *compensation_qubits]
        extra_compensation_qubits = [q for q in compensation_qubits if q not in [self.qubit_control, self.qubit_target]]
        
        for qubit in extra_compensation_qubits:
            qubit.align(self.qubit_control)
        self.transmon_pair.align()
                 
        pulse_duration = self.flux_pulse_control.length // 4 + 10
        
        for compensation in self.compensations:
            qubit = compensation["qubit"]
            # Assume amplitude is 100 mV by default
            qubit.z.play(f"const_100mV", duration=pulse_duration, amplitude_scale=compensation["shift"] / 0.1)
            frame_rotation_2pi(compensation["phase"], qubit.xy.name)
            qubit.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.qubit_control.z.wait(20)
        
        super().execute(*args, **kwargs)

        for qubit in extra_compensation_qubits:
            qubit.align(self.qubit_control)
        self.transmon_pair.align()

@quam_dataclass
class CNotGate_CT(TwoQubitGate):
    
    def execute(self):
        self.qubit_target.gates["Hadamard"]()
        self.qubit_pair.gates["Cz"]()
        self.qubit_target.gates["Hadamard"]()
        
@quam_dataclass
class CNotGate_TC(TwoQubitGate):
    def execute(self):
        self.qubit_control.gates["Hadamard"]()
        self.qubit_control.align(self.qubit_target)
        self.qubit_pair.gates["Cz"]()
        self.qubit_control.align(self.qubit_target)
        self.qubit_control.gates["Hadamard"]()
        
        
@quam_dataclass
class SWAP_Coupler_Gate(TwoQubitGate):
    """SWAP Operation for a qubit pair"""

    # Pulses will be added to qubit elements
    # The reason we don't add "flux_to_q1" directly to q1.z is because it is part of
    # the CZ operation, i.e. it is only applied as part of a CZ operation

    flux_pulse_control: Pulse
    coupler_pulse_control: Pulse

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0    
    pre_wait: int = 10
    post_wait: int = 20

    @property
    def gate_label(self) -> str:
        try:
            return self.parent.get_attr_name(self)
        except AttributeError:
            return "SWAP_Coupler_Gate"

    @property
    def flux_pulse_control_label(self) -> str:
        if self.flux_pulse_control.id is not None:
            pulse_label = self.flux_pulse_control.id
        else:
            pulse_label = "flux_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    @property
    def coupler_pulse_control_label(self) -> str:
        if self.coupler_pulse_control.id is not None:
            pulse_label = self.coupler_pulse_control.id
        else:
            pulse_label = "coupler_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    def execute(self, amplitude_scale=None, coupler_pulse_scale=None):        
        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )
        self.coupler.play(
            self.coupler_pulse_control_label,
            validate=False,
            amplitude_scale=coupler_pulse_scale,
        )
        
        self.transmon_pair.align()
        self.qubit_control.xy.wait(self.post_wait)
        self.transmon_pair.align()

    @property
    def config_settings(self):
        return {"after": [self.qubit_control.z]}

    def apply_to_config(self, config: dict) -> None:
        pulse_control = copy(self.flux_pulse_control)
        pulse_control.id = self.flux_pulse_control_label
        pulse_control.parent = None  # Reset parent so it can be attached to new parent
        pulse_control.parent = self.qubit_control.z
        
        pulse_coupler = copy(self.coupler_pulse_control)
        pulse_coupler.id = self.coupler_pulse_control_label
        pulse_coupler.parent = None  # Reset parent so it can be attached to new parent
        pulse_coupler.parent = self.coupler

        if self.flux_pulse_control_label in self.qubit_control.z.operations:
            raise ValueError(
                "Pulse name already exists in pulse operations. "
                f"Channel: {self.qubit_control.z.get_reference()}, "
                f"Pulse: {self.flux_pulse_control.get_reference()}, "
                f"Pulse name: {self.flux_pulse_control_label}"
            )

        pulse_control.apply_to_config(config)
        pulse_coupler.apply_to_config(config)

        element_config_control = config["elements"][self.qubit_control.z.name]
        element_config_control["operations"][self.flux_pulse_control_label] = pulse_control.pulse_name
        element_config_coupler = config["elements"][self.coupler.name]
        element_config_coupler["operations"][self.coupler_pulse_control_label] = pulse_coupler.pulse_name
