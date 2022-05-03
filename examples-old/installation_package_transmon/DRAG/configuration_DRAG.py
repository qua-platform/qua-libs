import numpy as np
from scipy.signal.windows import gaussian

# Definition of pulses follow Chen et al. PRL, 116, 020501 (2016)
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


def drag_cosine_pulse_waveforms(amplitude, length, alpha, detune, delta):
    t = np.arange(length, dtype=int)  # array of size pulse length in ns
    cos_wave = 0.5 * amplitude * (1 - np.cos(t * 2 * np.pi / length))  # cosine function
    sin_wave = 0.5 * amplitude * (2 * np.pi / length * 1e9) * np.sin(t * 2 * np.pi / length)  # derivative of cos_wave
    z = cos_wave + 1j * sin_wave * (alpha / delta)  # complex DRAG envelope
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


drag_len = 16  # length of pulse in ns
drag_amp = 0.1  # amplitude of pulse in Volts
drag_del_f = -0e6  # Detuning frequency in MHz
drag_alpha = 1  # DRAG coefficient
drag_delta = 2 * np.pi * (-200e6 - drag_del_f)  # Updated drag_delta, see Eqn. (4) in Chen et al.

# Definition of I- and Q-quadratures DRAG waveforms for pi and pi_half pulses with a gaussian envelope
drag_gauss_wf, drag_gauss_der_wf = drag_gaussian_pulse_waveforms(
    drag_amp,
    drag_len,
    drag_len / 5,
    drag_alpha,
    drag_del_f,
    drag_delta,
    substracted=False,
)  # pi pulse
drag_half_gauss_wf, drag_half_gauss_der_wf = drag_gaussian_pulse_waveforms(
    drag_amp * 0.5,
    drag_len,
    drag_len / 5,
    drag_alpha,
    drag_del_f,
    drag_delta,
    substracted=False,
)  # pi_half pulse
minus_drag_half_gauss_wf, minus_drag_half_gauss_der_wf = drag_gaussian_pulse_waveforms(
    drag_amp * (-0.5),
    drag_len,
    drag_len / 5,
    drag_alpha,
    drag_del_f,
    drag_delta,
    substracted=False,
)  # -pi_half pulse
minus_drag_gauss_wf, minus_drag_gauss_der_wf = drag_gaussian_pulse_waveforms(
    drag_amp * (-1),
    drag_len,
    drag_len / 5,
    drag_alpha,
    drag_del_f,
    drag_delta,
    substracted=False,
)  # -pi pulse

# Definition of I- and Q-quadratures DRAG waveforms for pi and pi_half pulses with a cosine envelope
drag_cos_wf, drag_sin_wf = drag_cosine_pulse_waveforms(
    drag_amp, drag_len, drag_alpha, drag_del_f, drag_delta
)  # pi pulse
drag_half_cos_wf, drag_half_sin_wf = drag_cosine_pulse_waveforms(
    drag_amp * 0.5, drag_len, drag_alpha, drag_del_f, drag_delta
)  # pi_half pulse
minus_drag_half_cos_wf, minus_drag_half_sin_wf = drag_cosine_pulse_waveforms(
    drag_amp * (-0.5), drag_len, drag_alpha, drag_del_f, drag_delta
)  # -pi_half pulse
minus_drag_cos_wf, minus_drag_sin_wf = drag_cosine_pulse_waveforms(
    drag_amp * (-1), drag_len, drag_alpha, drag_del_f, drag_delta
)  # -pi pulse

readout_len = 400  # length for readout pulse
qubit_IF = 0e6  # qubit intermediate frequency
rr_IF = 0  # resonator intermediate frequency
qubit_LO = 6.345e9  # LO for qubit drive
rr_LO = 4.755e9  # LO for resonator drive

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qubit 1-I
                2: {"offset": +0.0},  # qubit 1-Q
                # 3: {"offset": +0.0},  # Readout resonator
                # 4: {"offset": +0.0},  # Readout resonator
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
                "X/2_d": "Xpi_half_DRAG_pulse",
                "X_d": "Xpi_DRAG_pulse",
                "-X_d": "-Xpi_DRAG_pulse",
                "-X/2_d": "-Xpi_half_DRAG_pulse",
                "Y/2_d": "Ypi_half_DRAG_pulse",
                "Y_d": "Ypi_DRAG_pulse",
                "-Y_d": "-Ypi_DRAG_pulse",
                "-Y/2_d": "-Ypi_half_DRAG_pulse",
                "X/2_c": "Xpi_c_half_DRAG_pulse",
                "X_c": "Xpi_c_DRAG_pulse",
                "-X_c": "-Xpi_c_DRAG_pulse",
                "-X/2_c": "-Xpi_c_half_DRAG_pulse",
                "Y/2_c": "Ypi_c_half_DRAG_pulse",
                "Y_c": "Ypi_c_DRAG_pulse",
                "-Y_c": "-Ypi_c_DRAG_pulse",
                "-Y/2_c": "-Ypi_c_half_DRAG_pulse",
            },
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
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
        "Xpi_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_gauss_wf", "Q": "DRAG_gauss_der_wf"},
        },
        "-Xpi_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_gauss_wf", "Q": "-DRAG_gauss_der_wf"},
        },
        "Ypi_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_gauss_der_wf", "Q": "DRAG_gauss_wf"},
        },
        "-Ypi_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_gauss_der_wf", "Q": "-DRAG_gauss_wf"},
        },
        "Xpi_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_half_gauss_wf", "Q": "DRAG_half_gauss_der_wf"},
        },
        "-Xpi_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_half_gauss_wf", "Q": "-DRAG_half_gauss_der_wf"},
        },
        "Ypi_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_half_gauss_der_wf", "Q": "DRAG_half_gauss_wf"},
        },
        "-Ypi_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_half_gauss_der_wf", "Q": "-DRAG_half_gauss_wf"},
        },
        "Xpi_c_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_cos_wf", "Q": "DRAG_sin_wf"},
        },
        "-Xpi_c_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_cos_wf", "Q": "-DRAG_sin_wf"},
        },
        "Ypi_c_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_sin_wf", "Q": "DRAG_cos_wf"},
        },
        "-Ypi_c_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_sin_wf", "Q": "-DRAG_cos_wf"},
        },
        "Xpi_c_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_half_cos_wf", "Q": "DRAG_half_sin_wf"},
        },
        "-Xpi_c_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_half_cos_wf", "Q": "-DRAG_half_sin_wf"},
        },
        "Ypi_c_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "-DRAG_half_sin_wf", "Q": "DRAG_half_cos_wf"},
        },
        "-Ypi_c_half_DRAG_pulse": {
            "operation": "control",
            "length": drag_len,
            "waveforms": {"I": "DRAG_half_sin_wf", "Q": "-DRAG_half_cos_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": drag_len,
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "DRAG_gauss_wf": {"type": "arbitrary", "samples": drag_gauss_wf},
        "DRAG_gauss_der_wf": {"type": "arbitrary", "samples": drag_gauss_der_wf},
        "DRAG_half_gauss_wf": {"type": "arbitrary", "samples": drag_half_gauss_wf},
        "DRAG_half_gauss_der_wf": {
            "type": "arbitrary",
            "samples": drag_half_gauss_der_wf,
        },
        "-DRAG_half_gauss_wf": {
            "type": "arbitrary",
            "samples": minus_drag_half_gauss_wf,
        },
        "-DRAG_half_gauss_der_wf": {
            "type": "arbitrary",
            "samples": minus_drag_half_gauss_der_wf,
        },
        "-DRAG_gauss_wf": {"type": "arbitrary", "samples": minus_drag_gauss_wf},
        "-DRAG_gauss_der_wf": {"type": "arbitrary", "samples": minus_drag_gauss_der_wf},
        "DRAG_cos_wf": {"type": "arbitrary", "samples": drag_cos_wf},
        "DRAG_sin_wf": {"type": "arbitrary", "samples": drag_sin_wf},
        "DRAG_half_cos_wf": {"type": "arbitrary", "samples": drag_half_cos_wf},
        "DRAG_half_sin_wf": {"type": "arbitrary", "samples": drag_half_sin_wf},
        "-DRAG_half_cos_wf": {"type": "arbitrary", "samples": minus_drag_half_cos_wf},
        "-DRAG_half_sin_wf": {"type": "arbitrary", "samples": minus_drag_half_sin_wf},
        "-DRAG_cos_wf": {"type": "arbitrary", "samples": minus_drag_cos_wf},
        "-DRAG_sin_wf": {"type": "arbitrary", "samples": minus_drag_sin_wf},
        "readout_wf": {"type": "constant", "sample": 0.3},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "integW2": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
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
