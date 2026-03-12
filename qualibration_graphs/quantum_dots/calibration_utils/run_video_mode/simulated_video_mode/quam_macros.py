import numpy as np

from quam.core import quam_dataclass
from quam.components.macro import QubitMacro


@quam_dataclass
class MeasureMacro(QubitMacro):
    """Perform measurement on component."""

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def _setup(self):
        machine = self.qubit.machine

        # One preferred readout dot per qubit
        preferred_readout_dot = self.qubit.preferred_readout_quantum_dot

        # Preferred readout dot + qubit quantum dot form a qd pair
        qd_pair = machine.find_quantum_dot_pair(self.qubit.quantum_dot.id, preferred_readout_dot)

        self.sensors = [k.get_reference() for k in machine.quantum_dot_pairs[qd_pair].sensor_dots]

        for s in self.sensors:
            readout_pulse = s.readout_resonator.operations[self.pulse_name]
            readout_pulse.length = self.readout_duration

    def apply(self, **kwargs):
        machine = self.qubit.machine
        if not hasattr(self, "sensors"):
            self._setup()
        from qm.qua import save

        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables(num_IQ_pairs=len(self.sensors))

        for i, s in enumerate(self.sensors):
            I[i], Q[i] = s.readout_resonator.measure(self.pulse_name)
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])

        return I[0]  # Just a placeholder measure function. Probably not correct


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
