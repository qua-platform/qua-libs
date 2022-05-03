import numpy as np

gauss_len = 100
gaussian_amp = 0.2


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


delf = 0.0  # Detuning frequency e.g. [-25,-10] MHz
gauss_pulse = gauss(gaussian_amp, 0, 6, delf, gauss_len)
drag_gauss_pulse = gauss(gaussian_amp, 0, 6, delf, gauss_len)
alpha = 0.05
delta = 0.8 - 2 * np.pi * delf  # Below Eqn. (4) in Chen et al.
drag_gauss_der_pulse = gauss_der(alpha / delta * gaussian_amp, 0, 6, delf, gauss_len)

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
                3: {"offset": +0.0},  # qubit 2-I
                4: {"offset": +0.0},  # qubit 2-Q
                5: {"offset": +0.0},  # qubit 3-I
                6: {"offset": +0.0},  # qubit 3-Q
                7: {"offset": +0.0},  # qubit ancilla-I
                8: {"offset": +0.0},  # qubit ancilla-Q
                9: {"offset": +0.0},  # Readout resonator
                10: {"offset": +0.0},  # Readout resonator
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qubit 1-I
                2: {"offset": +0.0},  # qubit 1-Q
                3: {"offset": +0.0},  # qubit 2-I
                4: {"offset": +0.0},  # qubit 2-Q
                5: {"offset": +0.0},  # qubit 3-I
                6: {"offset": +0.0},  # qubit 3-Q
                7: {"offset": +0.0},  # qubit ancilla-I
                8: {"offset": +0.0},  # qubit ancilla-Q
                9: {"offset": +0.0},  # Readout resonator
                10: {"offset": +0.0},  # Readout resonator
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "q0": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "singleInput": {"port": ("con2", 1)},
            "intermediate_frequency": qubit_IF,
            "operations": {
                "X/2": "DRAG_PULSE",
                "X": "DRAG_PULSE",
                "-X/2": "DRAG_PULSE",
                "Y/2": "DRAG_PULSE",
                "Y": "DRAG_PULSE",
                "-Y/2": "DRAG_PULSE",
                "hadamard": "DRAG_PULSE",
            },
        },
        "q1": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
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
                "hadamard": "DRAG_PULSE",
            },
        },
        "q2": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
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
                "hadamard": "DRAG_PULSE",
            },
        },
        "q_anc": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
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
                "hadamard": "DRAG_PULSE",
            },
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
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
        "DRAG_gauss_wf": {"type": "arbitrary", "samples": drag_gauss_pulse},
        "DRAG_gauss_der_wf": {"type": "arbitrary", "samples": drag_gauss_der_pulse},
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
