from quam.core import quam_dataclass
from quam.components.pulses import Pulse
import numpy as np


@quam_dataclass
class DragPulseCosine(Pulse):
    """
    Creates Cosine based DRAG waveforms that compensate for the leakage and for the AC stark shift.

    These DRAG waveforms has been implemented following the next Refs.:
    Chen et al. PRL, 116, 020501 (2016)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
    and Chen's thesis
    https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
    :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
    :return: Returns a tuple of two lists. The first list is the I waveform (real part) and the second is the
        Q waveform (imaginary part)
    """

    axis_angle: float
    amplitude: float
    alpha: float
    anharmonicity: float
    detuning: float = 0.0
    subtracted: bool = True

    def waveform_function(self):
        from qualang_tools.config.waveform_tools import drag_cosine_pulse_waveforms

        I, Q = drag_cosine_pulse_waveforms(
            amplitude=self.amplitude,
            length=self.length,
            alpha=self.alpha,
            anharmonicity=self.anharmonicity,
            detuning=self.detuning,
            subtracted=self.subtracted,
        )
        I, Q = np.array(I), np.array(Q)

        I_rot = I * np.cos(self.axis_angle) - Q * np.sin(self.axis_angle)
        Q_rot = I * np.sin(self.axis_angle) + Q * np.cos(self.axis_angle)

        return I_rot + 1.0j * Q_rot

@quam_dataclass
class FluxPulse(Pulse):
    """Flux pulse QuAM component.

    Args:
        length (int): The total length of the pulse in samples, including zero padding.
        digital_marker (str, list, optional): The digital marker to use for the pulse.
        amplitude (float): The amplitude of the pulse in volts.
    """

    amplitude: float
    zero_padding: int = 0

    def waveform_function(self):
        waveform = self.amplitude * np.ones(self.length)
        if self.zero_padding:
            if self.zero_padding > self.length:
                raise ValueError(
                    f"Flux pulse zero padding ({self.zero_padding} ns) exceeds " f"pulse length ({self.length} ns)."
                )
            waveform[-self.zero_padding :] = 0
        return waveform
    
@quam_dataclass
class SNZPulse(Pulse):
    amplitude: float
    step_amplitude: float
    step_length: int
    spacing : int
    
    def __post_init__(self):
        self.length -= self.length % 4

    def waveform_function(self):
        rect_duration = (self.length - 4 - 2 * self.step_length - self.spacing) // 2
        waveform = [self.amplitude] * rect_duration
        waveform += [self.step_amplitude] * self.step_length
        waveform += [0] * self.spacing
        waveform += [-self.step_amplitude] * self.step_length
        waveform += [-self.amplitude] * rect_duration
        waveform += [0.0] * (self.length - len(waveform))
        
        return waveform    