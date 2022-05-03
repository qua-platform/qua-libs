import numpy as np

gauss_len = 32
gaussian_amp = 0.2


def square_gauss(amplitude, sigma, length):
    t = np.linspace(-10, 10, 20)
    gauss_wave = amplitude * np.exp(-(t**2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave[:10]] + [amplitude] * length + [float(x) for x in gauss_wave[10:]]


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]


square_gauss_pulse = square_gauss(gaussian_amp, 6, gauss_len)
gauss_len += 20

readout_len = 100

qubit_IF = 80e6
qubit_LO = 4.2e9

ge_IF = 80e6
ef_IF = 10e6

rr_IF = 55e6
rr_LO = 5.2e9

flux_IF = 300e6
flux_LO = 2.2e9

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
        "charge_line": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "pi_g_e": "gePulse",
                "pi_e_f": "efPulse",
                "X": "XPulse",
                "Y": "YPulse",
                "X/2": "X/2Pulse",
                "Y/2": "Y/2Pulse",
                "-X/2": "-X/2Pulse",
                "-Y/2": "-Y/2Pulse",
            },
        },
        "flux_line": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": flux_LO,
                "mixer": "mixer_flux",
            },
            "intermediate_frequency": flux_IF,
            "operations": {
                "iswap": "iswapPulse",
                "modulate": "iswapPulse",
            },
        },
        "readout_resonator": {
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
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "iswapPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "square_gauss_wf", "Q": "zero_wf"},
        },
        "XPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "X/2Pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi/2_wf", "Q": "zero_wf"},
        },
        "-X/2Pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "-pi/2_wf", "Q": "zero_wf"},
        },
        "YPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "zero_wf", "Q": "pi_wf"},
        },
        "Y/2Pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "zero_wf", "Q": "pi/2_wf"},
        },
        "-Y/2Pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "zero_wf", "Q": "-pi/2_wf"},
        },
        "gePulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi_wf", "Q": "pi_wf"},
        },
        "efPulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi_wf", "Q": "pi_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": gauss_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW_cos": "integW1",
                "integW_sin": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "square_gauss_wf": {"type": "arbitrary", "samples": square_gauss_pulse},
        "readout_wf": {"type": "constant", "sample": 0.3},
        "pi_wf": {"type": "arbitrary", "samples": gauss(0.2, 0, 12, gauss_len)},
        "-pi/2_wf": {"type": "arbitrary", "samples": gauss(-0.1, 0, 12, gauss_len)},
        "pi/2_wf": {"type": "arbitrary", "samples": gauss(0.1, 0, 12, gauss_len)},
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
                "correction": [1, 0, 0, 1],
            }
        ],
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": [1, 0, 0, 1],
            }
        ],
        "mixer_flux": [
            {
                "intermediate_frequency": flux_IF,
                "lo_frequency": flux_LO,
                "correction": [1, 0, 0, 1],
            }
        ],
    },
}
