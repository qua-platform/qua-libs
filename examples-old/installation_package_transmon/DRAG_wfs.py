import numpy as np
from scipy.signal.windows import gaussian


# Definition of pulses follow Chen et al. PRL, 116, 020501 (2016)


# I-quadrature component for gaussian shaped pulses
def gaussian_detuned(amplitude, sigma, delf, length, alph, delt):
    t = np.linspace(0, length, length)
    gauss_wave = amplitude * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))
    gauss_der_wave = (
        amplitude
        * (-2 * 1e9 * (t - length / 2) / (2 * sigma**2))
        * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))
    )
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gaussian_detuned_wave = gauss_wave * np.cos(2 * np.pi * delf * t * 1e-9) - (alph / delt) * gauss_der_wave * np.sin(
        2 * np.pi * delf * t * 1e-9
    )
    return [float(x) for x in gaussian_detuned_wave]


# Q-quadrature component for gaussian shaped pulses
def gaussian_derivative_detuned(amplitude, sigma, delf, length, alph, delt):
    t = np.linspace(0, length, length)
    gauss_wave = amplitude * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))
    gauss_der_wave = (
        amplitude
        * (-2 * 1e9 * (t - length / 2) / (2 * sigma**2))
        * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))
    )
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gaussian_derivative_detuned_wave = (alph / delt) * gauss_der_wave * np.cos(
        2 * np.pi * delf * t * 1e-9
    ) + gauss_wave * np.sin(2 * np.pi * delf * t * 1e-9)
    return [float(x) for x in gaussian_derivative_detuned_wave]


# I-quadrature component for cosine shaped pulses
def cos_detuned(amplitude, delf, length, alph, delt):
    t = np.linspace(0, length, length)
    cos_wave = 0.5 * amplitude * (1 - np.cos(t * 2 * np.pi / length))
    sin_wave = 0.5 * amplitude * (2 * np.pi / length * 1e9) * np.sin(t * 2 * np.pi / length)
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    cos_wave = cos_wave * np.cos(2 * np.pi * delf * t * 1e-9) - (alph / delt) * sin_wave * np.sin(
        2 * np.pi * delf * t * 1e-9
    )
    return [float(x) for x in cos_wave]


# Q-quadrature component for cosine shaped pulses
def cos_derivative_detuned(amplitude, delf, length, alph, delt):
    t = np.linspace(0, length, length)
    cos_wave = 0.5 * amplitude * (1 - np.cos(t * 2 * np.pi / length))
    sin_wave = 0.5 * amplitude * (2 * np.pi / length * 1e9) * np.sin(t * 2 * np.pi / length)
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    cos_derivative_detuned_wave = (alph / delt) * sin_wave * np.cos(2 * np.pi * delf * t * 1e-9) + cos_wave * np.sin(
        2 * np.pi * delf * t * 1e-9
    )
    return [float(x) for x in cos_derivative_detuned_wave]


# IQ imbalance function
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


drag_len = 16  # length of pulse in ns
drag_amp = 0.1  # amplitude of pulse in Volts
del_f = -0e6  # Detuning frequency in MHz
alpha = 1  # DRAG coefficient
delta = 2 * np.pi * (-200e6 - del_f)  # Updated Delta, see Eqn. (4) in Chen et al.

gauss_wf = gaussian_detuned(drag_amp, drag_len / 5, del_f, drag_len, alpha, delta)

# Definition of I-quadrature DRAG waveforms for pi and pi_half pulses with a gaussian envelope
drag_gauss_wf = gaussian_detuned(drag_amp, drag_len / 5, del_f, drag_len, alpha, delta)  # pi pulse
drag_half_gauss_wf = gaussian_detuned(drag_amp * 0.5, drag_len / 5, del_f, drag_len, alpha, delta)  # pi_half pulse
minus_drag_half_gauss_wf = gaussian_detuned(
    drag_amp * (-0.5), drag_len / 5, del_f, drag_len, alpha, delta
)  # -pi_half pulse
minus_drag_gauss_wf = gaussian_detuned(drag_amp * (-1), drag_len / 5, del_f, drag_len, alpha, delta)  # -pi pulse

# Definition of Q-quadrature DRAG waveforms for pi and pi_half pulses with a gaussian envelope
drag_gauss_der_wf = gaussian_derivative_detuned(drag_amp, drag_len / 5, del_f, drag_len, alpha, delta)  # pi pulse
drag_half_gauss_der_wf = gaussian_derivative_detuned(
    drag_amp * 0.5, drag_len / 5, del_f, drag_len, alpha, delta
)  # pi_half pulse
minus_drag_half_gauss_der_wf = gaussian_derivative_detuned(
    drag_amp * (-0.5), drag_len / 5, del_f, drag_len, alpha, delta
)  # -pi_half pulse
minus_drag_gauss_der_wf = gaussian_derivative_detuned(
    drag_amp * (-1), drag_len / 5, del_f, drag_len, alpha, delta
)  # -pi pulse

# Definition of I-quadrature DRAG waveforms for pi and pi_half pulses with a cosine envelope
drag_cos_wf = cos_detuned(drag_amp, del_f, drag_len, alpha, delta)  # pi pulse
drag_half_cos_wf = cos_detuned(drag_amp * 0.5, del_f, drag_len, alpha, delta)  # pi_half pulse
minus_drag_half_cos_wf = cos_detuned(drag_amp * (-0.5), del_f, drag_len, alpha, delta)  # -pi_half pulse
minus_drag_cos_wf = cos_detuned(drag_amp * (-1), del_f, drag_len, alpha, delta)  # -pi pulse

# Definition of Q-quadrature DRAG waveforms for pi and pi_half pulses with a cosine envelope
drag_sin_wf = cos_derivative_detuned(drag_amp, del_f, drag_len, alpha, delta)  # pi pulse
drag_half_sin_wf = cos_derivative_detuned(drag_amp * 0.5, del_f, drag_len, alpha, delta)  # pi_half pulse
minus_drag_half_sin_wf = cos_derivative_detuned(drag_amp * (-0.5), del_f, drag_len, alpha, delta)  # -pi_half pulse
minus_drag_sin_wf = cos_derivative_detuned(drag_amp * (-1), del_f, drag_len, alpha, delta)  # -pi pulse
