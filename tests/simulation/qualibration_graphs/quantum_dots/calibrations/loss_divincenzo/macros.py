"""Custom macros for programmatic QuAM construction in simulation tests.

All physical gates derive from a single calibrated X180 Gaussian pulse:
  - X(theta): play x180 with amplitude_scale = theta / 180
  - Y(theta): frame shift +90 deg, play X(theta), frame shift -90 deg
  - Z(theta): virtual frame rotation by theta degrees
"""

from __future__ import annotations

from typing import Optional

from qm.qua import assign, declare, fixed, frame_rotation_2pi

from quam.core import quam_dataclass  # type: ignore[import-not-found]
from quam.core.macro.quam_macro import QuamMacro  # type: ignore[import-not-found]
from quam.utils.qua_types import QuaVariableBool  # type: ignore[import-not-found]


@quam_dataclass
class XGateMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """X-axis rotation derived from a single calibrated pi-pulse.

    amplitude_scale = theta / 180, so:
      - x(180)  → full amplitude  (pi pulse)
      - x(90)   → half amplitude  (pi/2 pulse)
      - x(-90)  → negative half   (-pi/2 pulse)
    """

    pulse_name: str = "x180"
    theta: float = 180.0
    duration: Optional[int] = None

    def apply(self, theta: Optional[float] = None, duration: Optional[int] = None, **kwargs) -> None:
        theta = theta if theta is not None else self.theta
        duration = duration if duration is not None else self.duration
        parent_qubit = self.parent.parent
        amplitude_scale = theta / 180.0
        parent_qubit.xy.play(
            self.pulse_name,
            amplitude_scale=amplitude_scale,
            duration=duration,
        )


@quam_dataclass
class YGateMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Y-axis rotation: frame shift +90 deg, play X(theta), frame shift -90 deg.

    Uses the same calibrated pi-pulse as XGateMacro, selecting the Y axis
    via an IQ frame rotation.
    """

    pulse_name: str = "x180"
    theta: float = 180.0
    duration: Optional[int] = None

    def apply(self, theta: Optional[float] = None, duration: Optional[int] = None, **kwargs) -> None:
        theta = theta if theta is not None else self.theta
        duration = duration if duration is not None else self.duration
        parent_qubit = self.parent.parent
        xy = parent_qubit.xy
        amplitude_scale = theta / 180.0
        frame_rotation_2pi(0.25, xy.name)
        xy.play(
            self.pulse_name,
            amplitude_scale=amplitude_scale,
            duration=duration,
        )
        frame_rotation_2pi(-0.25, xy.name)


@quam_dataclass
class ZGateMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Z-axis rotation: virtual frame rotation (zero hardware duration).

    frame_rotation_2pi(theta / 360) applies an R_z(theta) rotation.
    """

    theta: float = 180.0

    def apply(self, theta: Optional[float] = None, **kwargs) -> None:
        theta = theta if theta is not None else self.theta
        parent_qubit = self.parent.parent
        frame_rotation_2pi(theta / 360.0, parent_qubit.xy.name)


@quam_dataclass
class MeasureMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Measurement with integrated voltage point navigation and thresholding."""

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def _validate(self, parent_qubit) -> None:
        if parent_qubit.quantum_dot is None:
            raise ValueError("Cannot measure: quantum_dot is not configured on parent qubit.")

        if parent_qubit.preferred_readout_quantum_dot is None:
            raise ValueError("Cannot measure: preferred_readout_quantum_dot is not set on parent qubit.")

        pair_id = parent_qubit.machine.find_quantum_dot_pair(
            parent_qubit.quantum_dot.id, parent_qubit.preferred_readout_quantum_dot
        )
        pair = parent_qubit.machine.quantum_dot_pairs[pair_id]

        if not pair.sensor_dots:
            raise ValueError("Cannot measure: no sensor_dots configured on quantum dot pair.")

        sensor_dot = pair.sensor_dots[0]

        if sensor_dot.readout_resonator is None:
            raise ValueError("Cannot measure: readout_resonator is not configured on sensor_dot.")

    def apply(self, *args, **kwargs) -> QuaVariableBool:
        """Execute measurement sequence and return qubit state (parity)."""
        pulse = kwargs.get("pulse_name", self.pulse_name)
        duration = kwargs.get("duration", self.readout_duration)

        parent_qubit = self.parent.parent
        self._validate(parent_qubit)

        parent_qubit.step_to_point("measure", duration=duration)

        qd_pair_id = parent_qubit.machine.find_quantum_dot_pair(
            parent_qubit.quantum_dot.id, parent_qubit.preferred_readout_quantum_dot
        )
        pair = parent_qubit.machine.quantum_dot_pairs[qd_pair_id]
        sensor_dot = pair.sensor_dots[0]

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
