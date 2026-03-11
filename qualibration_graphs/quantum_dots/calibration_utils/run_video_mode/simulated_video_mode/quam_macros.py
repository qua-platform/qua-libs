import numpy as np

from quam_builder.architecture.quantum_dots.operations.default_macros import SINGLE_QUBIT_MACROS
from quam.core import quam_dataclass
from quam.components.macro import QubitMacro


@quam_dataclass
class MeasureMacro(QubitMacro):
    """Perform measurement on component."""

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def apply(self, **kwargs):
        machine = self.qubit.machine
        preferred_readout_dot = self.qubit.preferred_readout_quantum_dot
        qd_pair = machine.find_quantum_dot_pair(self.qubit.id, preferred_readout_dot)
        sensors = qd_pair.sensor_dots

        from qm.qua import save

        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables(num_IQ_pairs=len(sensors))

        for i, s in enumerate(sensors):
            I[i], Q[i] = s.readout_resonator.measure(self.pulse_name, duration=self.readout_duration)
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])

        return I, Q  # Just a placeholder measure function. Probably not correct


@quam_dataclass
class XGateMacro(QubitMacro):
    """Perform X gate on component."""

    amplitude_scale: float
    duration: int
    pulse_name: str = "x180"

    def apply(self, **kwargs):
        self.qubit.xy.play(
            self.pulse_name,
            amplitude_scale=self.amplitude_scale,
            duration=self.duration,
        )


@quam_dataclass
class YGateMacro(QubitMacro):
    """Perform X gate on component."""

    amplitude_scale: float
    duration: int
    pulse_name: str = "y180"

    def apply(self, **kwargs):
        self.qubit.xy.play(
            self.pulse_name,
            amplitude_scale=self.amplitude_scale,
            duration=self.duration,
        )


@quam_dataclass
class ZGateMacro(QubitMacro):
    """Perform X gate on component."""

    theta: float

    def apply(self, **kwargs):
        self.qubit.xy.frame_rotation(np.deg2rad(self.theta))
