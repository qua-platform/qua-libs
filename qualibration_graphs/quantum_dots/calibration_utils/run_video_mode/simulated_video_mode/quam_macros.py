"""QuAM macros for the simulated video-mode qubit state."""

from __future__ import annotations

from typing import Optional

from qm.qua import assign, declare, fixed, frame_rotation_2pi

from quam.core import quam_dataclass
from quam.core.macro.quam_macro import QuamMacro
from quam.utils.qua_types import QuaVariableBool


@quam_dataclass
class XGateMacro(QuamMacro):
    """Apply an X-axis rotation using the qubit XY pulse."""

    pulse_name: str = "x180"
    amplitude_scale: Optional[float] = 1.0
    duration: Optional[int] = 25

    def apply(self, *args, **kwargs) -> None:
        parent_qubit = self.parent.parent
        duration = kwargs.pop("duration", None)
        if duration is None and args:
            duration = args[0]
        if duration is None:
            duration = self.duration
        amplitude_scale = kwargs.get("amplitude_scale", self.amplitude_scale)

        if parent_qubit.xy is None:
            raise ValueError("Cannot apply X gate: xy is not configured on parent qubit.")
        if duration is None or amplitude_scale is None:
            raise ValueError("Cannot apply X gate: missing duration or amplitude scale.")

        parent_qubit.xy.play(
            self.pulse_name,
            amplitude_scale=amplitude_scale,
            duration=duration,
        )
        parent_qubit.voltage_sequence.step_to_voltages({}, duration=duration * 4)


@quam_dataclass
class YGateMacro(QuamMacro):
    """Apply a Y-axis rotation using a frame shift and the X pulse."""

    pulse_name: str = "x180"
    amplitude_scale: Optional[float] = 1.0
    duration: Optional[int] = 25

    def apply(self, *args, **kwargs) -> None:
        parent_qubit = self.parent.parent
        duration = kwargs.pop("duration", None)
        if duration is None and args:
            duration = args[0]
        if duration is None:
            duration = self.duration
        amplitude_scale = kwargs.get("amplitude_scale", self.amplitude_scale)

        if parent_qubit.xy is None:
            raise ValueError("Cannot apply Y gate: xy is not configured on parent qubit.")
        if duration is None or amplitude_scale is None:
            raise ValueError("Cannot apply Y gate: missing duration or amplitude scale.")

        frame_rotation_2pi(0.25, parent_qubit.xy.name)
        parent_qubit.xy.play(
            self.pulse_name,
            amplitude_scale=amplitude_scale,
            duration=duration,
        )
        parent_qubit.voltage_sequence.step_to_voltages({}, duration=duration * 4)
        frame_rotation_2pi(-0.25, parent_qubit.xy.name)


@quam_dataclass
class ZGateMacro(QuamMacro):
    """Apply a virtual Z rotation."""

    theta: float = 180.0

    def apply(self, theta: Optional[float] = None, **kwargs) -> None:
        del kwargs
        parent_qubit = self.parent.parent
        if parent_qubit.xy is None:
            raise ValueError("Cannot apply Z gate: xy is not configured on parent qubit.")

        angle = theta if theta is not None else self.theta
        frame_rotation_2pi(angle / 360.0, parent_qubit.xy.name)


@quam_dataclass
class MeasureMacro(QuamMacro):
    """Measure the qubit using the configured parity-readout sensor."""

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def _get_qubit_pair(self, parent_qubit):
        preferred_readout_dot = getattr(parent_qubit, "preferred_readout_quantum_dot", None)

        for pair_id, pair in parent_qubit.machine.quantum_dot_pairs.items():
            dot_ids = {dot.id for dot in pair.quantum_dots}
            if parent_qubit.quantum_dot.id not in dot_ids:
                continue
            if preferred_readout_dot is not None and preferred_readout_dot not in dot_ids:
                continue
            if pair.sensor_dots:
                return pair_id, pair

        raise ValueError("Cannot measure: no suitable quantum dot pair with sensor readout was found.")

    def _validate(self, parent_qubit) -> None:
        if parent_qubit.quantum_dot is None:
            raise ValueError("Cannot measure: quantum_dot is not configured on parent qubit.")
        _, pair = self._get_qubit_pair(parent_qubit)
        if not pair.sensor_dots:
            raise ValueError("Cannot measure: no sensor dots configured on the quantum dot pair.")

        sensor_dot = pair.sensor_dots[0]
        if sensor_dot.readout_resonator is None:
            raise ValueError("Cannot measure: readout resonator is not configured on the sensor dot.")

    def apply(self, *args, **kwargs) -> QuaVariableBool:
        del args
        pulse_name = kwargs.get("pulse_name", self.pulse_name)
        duration = kwargs.get("duration", self.readout_duration)

        parent_qubit = self.parent.parent
        self._validate(parent_qubit)

        parent_qubit.step_to_point("measure", duration=duration)

        pair_id, pair = self._get_qubit_pair(parent_qubit)
        sensor_dot = pair.sensor_dots[0]

        i_var = declare(fixed)
        q_var = declare(fixed)

        sensor_dot.readout_resonator.wait(64)
        sensor_dot.readout_resonator.measure(
            pulse_name,
            qua_vars=(i_var, q_var),
        )

        threshold = sensor_dot.readout_thresholds.get(pair_id, 0.0)
        state = declare(bool)
        assign(state, i_var > threshold)
        return state
