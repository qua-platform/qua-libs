"""Custom macros for programmatic QuAM construction in simulation tests."""

from __future__ import annotations

from typing import Optional

from qm.qua import assign, declare, fixed

from quam.core import quam_dataclass  # type: ignore[import-not-found]
from quam.core.macro.quam_macro import QuamMacro  # type: ignore[import-not-found]
from quam.utils.qua_types import QuaVariableBool  # type: ignore[import-not-found]


@quam_dataclass
class X180Macro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Macro for X180 gate: step to operate point and apply pi pulse."""

    pulse_name: str = "X180"
    amplitude_scale: Optional[float] = None
    duration: Optional[int] = None

    def _validate(self, xy_channel, duration, amplitude_scale) -> None:
        if xy_channel is None:
            raise ValueError(
                "Cannot apply X180 gate: xy_channel is not configured on parent qubit."
            )

        missing = []
        if duration is None:
            missing.append("duration")
        if amplitude_scale is None:
            missing.append("amplitude_scale")
        if missing:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}. "
                "Provide via kwargs or set as class attributes."
            )

    def apply(self, *args, **kwargs) -> None:
        """Execute X180 gate sequence."""
        parent_qubit = self.parent.parent
        amp_scale = kwargs.get("amplitude_scale", self.amplitude_scale)
        duration = kwargs.pop("duration", None)
        if duration is None and args:
            duration = args[0]
        if duration is None:
            duration = self.duration

        self._validate(parent_qubit.xy_channel, duration, amp_scale)

        parent_qubit.xy_channel.play(
            self.pulse_name,
            amplitude_scale=amp_scale,
            duration=duration,
        )


@quam_dataclass
class MeasureMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Macro for measurement with integrated voltage point navigation and thresholding."""

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def _validate(self, parent_qubit) -> None:
        if not parent_qubit.sensor_dots:
            raise ValueError("Cannot measure: no sensor_dots configured on parent qubit.")

        sensor_dot = parent_qubit.sensor_dots[0]

        if sensor_dot.readout_resonator is None:
            raise ValueError("Cannot measure: readout_resonator is not configured on sensor_dot.")

        if parent_qubit.quantum_dot is None:
            raise ValueError("Cannot measure: quantum_dot is not configured on parent qubit.")

        if parent_qubit.preferred_readout_quantum_dot is None:
            raise ValueError(
                "Cannot measure: preferred_readout_quantum_dot is not set on parent qubit."
            )

    def apply(self, *args, **kwargs) -> QuaVariableBool:
        """Execute measurement sequence and return qubit state (parity)."""
        pulse = kwargs.get("pulse_name", self.pulse_name)

        parent_qubit = self.parent.parent
        self._validate(parent_qubit)

        parent_qubit.step_to_point("measure", duration=self.readout_duration)

        sensor_dot = parent_qubit.sensor_dots[0]

        qd_pair_id = parent_qubit.machine.find_quantum_dot_pair(
            parent_qubit.quantum_dot.id, parent_qubit.preferred_readout_quantum_dot
        )

        I = declare(fixed)
        Q = declare(fixed)

        sensor_dot.readout_resonator.wait(64)
        sensor_dot.readout_resonator.measure(
            pulse,
            qua_vars=(I, Q),
        )

        threshold = sensor_dot.readout_thresholds.get(qd_pair_id, 0.0)

        state = declare(bool)
        assign(state, I > threshold)

        return state
