import numpy as np

gauss_len = 16  # pulse length in ns
gaussian_amp = 0.2  # pulse amplitude in volts

# Definition of pulses follow Chen et al. PRL, 116, 020501 (2016)


# I-quadrature component for gaussian shaped pulses
def gauss(amplitude, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-(t**2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]


# Q-quadrature component for gaussian shaped pulses
def gauss_derivative(amplitude, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_derivative_wave = amplitude * (-2 * 1e9 * t / (2 * sigma**2)) * np.exp(-(t**2) / (2 * sigma**2))
    return [float(x) for x in gauss_derivative_wave]


# Definition of I- and Q-quadratures DRAG waveforms for pi and pi_half pulses with a gaussian envelope
def drag_gaussian_pulse_waveforms(amplitude, length, sigma, alpha, detune, delta, substracted):
    t = np.arange(length, dtype=int)  # array of size pulse length in ns
    gauss_wave = amplitude * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))  # gaussian function
    gauss_der_wave = (
        amplitude
        * (-2 * 1e9 * (t - length / 2) / (2 * sigma**2))
        * np.exp(-((t - length / 2) ** 2) / (2 * sigma**2))
    )  # derivative of gaussian
    if substracted:  # if statement to determine usage of subtracted gaussian
        gauss_wave = gauss_wave - gauss_wave[-1]  # subtracted gaussian
    z = gauss_wave + 1j * gauss_der_wave * (alpha / delta)  # complex DRAG envelope
    z *= np.exp(1j * 2 * np.pi * detune * t * 1e-9)  # complex DRAG detuned envelope
    I_wf = z.real.tolist()  # get the real part of the I component of waveform
    Q_wf = z.imag.tolist()  # get the real part of the Q component of waveform
    return I_wf, Q_wf


# IQ imbalance function for mixer calibration
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


# Parameters needed to define DRAG pulses
del_f = -0.0e6  # Detuning frequency -10 MHz
drag_alpha = 0.5  # DRAG coefficient
drag_delta = 2 * np.pi * (-200e6 - del_f)  # Below Eqn. (4) in Chen et al.

# DRAG waveforms
gauss_pulse = gauss(gaussian_amp, gauss_len / 5, gauss_len)
drag_I_wf, drag_Q_wf = drag_gaussian_pulse_waveforms(
    gaussian_amp,
    gauss_len,
    gauss_len / 5,
    drag_alpha,
    del_f,
    drag_delta,
    substracted=False,
)

readout_len = 400
qubit_IF = 0
rr_IF = 0
qubit_LO = 6.345e9
rr_LO = 4.755e9

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qubit 1-I
                2: {"offset": +0.0},  # qubit 1-Q
                3: {"offset": +0.0},  # Readout resonator
                4: {"offset": +0.0},  # Readout resonator
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X/2": "DRAG_PULSE",
                "X": "DRAG_PULSE",
                "-X/2": "DRAG_PULSE",
                "Y/2": "DRAG_PULSE",
                "Y": "DRAG_PULSE",
                "-Y/2": "DRAG_PULSE",
            },
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "XPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "YPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "zero_wf", "Q": "gauss_wf"},
        },
        "DRAG_PULSE": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "DRAG_gauss_wf", "Q": "DRAG_gauss_der_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": gauss_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_pulse},
        "DRAG_gauss_wf": {"type": "arbitrary", "samples": drag_I_wf},
        "DRAG_gauss_der_wf": {
            "type": "arbitrary",
            "samples": drag_Q_wf,
        },
        "readout_wf": {"type": "constant", "sample": 0.3},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "integW2": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
    },
}
