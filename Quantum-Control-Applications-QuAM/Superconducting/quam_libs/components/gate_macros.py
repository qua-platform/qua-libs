from typing import Union, Literal

import numpy as np
from quam.components.macro import QubitMacro, QubitPairMacro, PulseMacro
from quam.components.pulses import ReadoutPulse, Pulse
from quam.core import quam_dataclass
from .transmon import Transmon
from .tunable_coupler import TunableCoupler
from .readout_resonator import ReadoutResonatorIQ
from qm.qua import declare, assign, fixed
from quam.utils.qua_types import QuaVariableBool

__all__ = ["MeasureMacro", "ResetMacro", "VirtualZMacro", "CZMacro", "DelayMacro", "IdMacro"]


def get_pulse_name(pulse: Pulse) -> str:
    """
    Get the name of the pulse. If the pulse has an id, return it.
    """
    if pulse.id is not None:
        return pulse.id
    elif pulse.parent is not None:
        return pulse.parent.get_attr_name(pulse)
    else:
        raise AttributeError(f"Cannot infer id of {pulse} because it is not attached to a parent")


@quam_dataclass
class MeasureMacro(QubitMacro):
    pulse: Union[ReadoutPulse, str] = "readout"

    def apply(self, **kwargs) -> QuaVariableBool:
        state: QuaVariableBool = kwargs.get("state", declare(bool))
        qua_vars = kwargs.get("qua_vars", (declare(fixed), declare(fixed)))
        pulse: ReadoutPulse = (
            self.pulse if isinstance(self.pulse, Pulse) else self.qubit.get_pulse(self.pulse)
        )

        resonator: ReadoutResonatorIQ = self.qubit.resonator
        resonator.measure(get_pulse_name(pulse), qua_vars=qua_vars)
        I, Q = qua_vars
        assign(state, I > pulse.threshold)
        return state


@quam_dataclass
class ResetMacro(QubitMacro):
    reset_type: Literal["active", "thermalize"] = "active"
    pi_pulse: Union[Pulse, str] = "x"
    readout_pulse: Union[ReadoutPulse, str] = "measure"
    max_attempts: int = 5
    thermalize_time: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.max_attempts > 0, "max_attempts must be greater than 0"

    def apply(self, **kwargs) -> None:
        pi_pulse: Pulse = (
            self.pi_pulse
            if isinstance(self.pi_pulse, Pulse)
            else self.qubit.get_pulse(self.pi_pulse)
        )
        readout_pulse: ReadoutPulse = (
            self.readout_pulse
            if isinstance(self.readout_pulse, Pulse)
            else self.qubit.get_pulse(self.readout_pulse)
        )
        if self.reset_type == "active":
            from ..macros import active_reset
            active_reset(self.qubit, pi_pulse_name=get_pulse_name(pi_pulse),
                         readout_pulse_name=get_pulse_name(readout_pulse),
                         max_attempts=self.max_attempts)
            # self.qubit.reset_qubit_active(
            #     pi_pulse_name=get_pulse_name(pi_pulse),
            #     readout_pulse_name=get_pulse_name(readout_pulse),
            #     max_attempts=self.max_attempts,
            # )
        else:
            # Thermalize the qubit
            self.qubit.wait(self.thermalize_time // 4)


@quam_dataclass
class VirtualZMacro(QubitMacro):
    def apply(self, angle: float) -> None:
        self.qubit.xy.frame_rotation(angle)


@quam_dataclass
class CZMacro(QubitPairMacro):
    flux_pulse_control: Union[Pulse, str]
    coupler_flux_pulse: Pulse = None

    pre_wait: int = 4

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0
    
    @property
    def coupler(self) -> TunableCoupler:
        return self.qubit_pair.coupler
    
    @property
    def flux_pulse_control_label(self) -> str:
        pulse = (
            self.qubit_control.get_pulse(self.flux_pulse_control)
            if isinstance(self.flux_pulse_control, str)
            else self.flux_pulse_control
        )
        return get_pulse_name(pulse)

    @property
    def coupler_flux_pulse_label(self) -> str:
        pulse = (
            self.coupler.get_pulse(self.coupler_flux_pulse)
            if isinstance(self.coupler_flux_pulse, str)
            else self.coupler_flux_pulse
        )
        return get_pulse_name(pulse)

    def apply(
        self,
        *,
        amplitude_scale=None,
        phase_shift_control=None,
        phase_shift_target=None,
        **kwargs,
    ) -> None:
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )

        if self.coupler_flux_pulse is not None:
            self.qubit_pair.coupler.play(self.coupler_flux_pulse_label, validate=False)

        self.qubit_pair.align()
        if phase_shift_control is not None:
            self.qubit_control.xy.frame_rotation_2pi(phase_shift_control)
        elif np.abs(self.phase_shift_control) > 1e-6:
            self.qubit_control.xy.frame_rotation_2pi(self.phase_shift_control)
        if phase_shift_target is not None:
            self.qubit_target.xy.frame_rotation_2pi(phase_shift_target)
        elif np.abs(self.phase_shift_target) > 1e-6:
            self.qubit_target.xy.frame_rotation_2pi(self.phase_shift_target)

        self.qubit_control.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.qubit_target.xy.play("x180", amplitude_scale=0.0, duration=4)
        self.qubit_pair.align()

# this is a place holder for now 
@quam_dataclass
class CZFixedMacro(QubitPairMacro):
    flux_pulse_control: Union[Pulse, str] = None
    
    def apply(
        self,
        *,
        amplitude_scale=None,
        phase_shift_control=None,
        phase_shift_target=None,
        **kwargs,
    ) -> None:
        self.qubit_control.wait(1000//4)


@quam_dataclass
class DelayMacro(QubitMacro):

    def apply(self, duration) -> None:
        qubit: Transmon = self.qubit
        qubit.wait(duration)

@quam_dataclass
class IdMacro(QubitMacro):
    """
    Identity macro for a qubit.
    This macro does not perform any operation on the qubit.
    It is used to ensure that the qubit is in a valid state.
    """

    def apply(self, **kwargs) -> None:
        # No operation is performed
        self.qubit.align()

@quam_dataclass
class HadamardGate(QubitMacro):
    """single qubit Hadamard gate

    """

    def apply(self):
        self.qubit.xy.play('y90')
        self.qubit.xy.play('x180')


@quam_dataclass
class XGate(QubitMacro):
    def apply(self, *args, **kwargs):
        self.qubit.xy.play('x180')


@quam_dataclass
class YGate(QubitMacro):
    def apply(self, *args, **kwargs):
        self.qubit.xy.play('y180')


@quam_dataclass
class SXGate(QubitMacro):
    def apply(self, *args, **kwargs):
        self.qubit.xy.play('x90')


@quam_dataclass
class UGate(QubitMacro):
    def _rz(self, angle):
        self.qubit.xy.frame_rotation(angle)

    def _x90(self):
        self.qubit.xy.play('x90')

    # implementation based on https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.U3Gate
    def apply(self, *args, **kwargs):
        theta = args[0]
        phi = args[1]
        lambda_ = args[2]
        self._rz(lambda_)
        self._x90()
        self._rz(theta + np.pi)
        self._x90()
        self._rz(phi + np.pi)


@quam_dataclass
class U1Gate(UGate):
    def apply(self, *args, **kwargs):
        return super().apply(0, 0, args[0])


@quam_dataclass
class U3Gate(UGate):
    pass
