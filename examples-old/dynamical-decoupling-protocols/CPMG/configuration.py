import numpy as np

pulse_len = 100
readout_len = 400
qubit_IF = 50e6
rr_IF = 50e6
qubit_LO = 6.345e9
rr_LO = 4.755e9


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


gauss_pulse = gauss(0.2, 0, 12, pulse_len)

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qe-I
                2: {"offset": +0.0},  # qe-Q
                3: {"offset": +0.0},  # rr-I
                4: {"offset": +0.0},  # rr-Q
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
            },
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 0,
            "operations": {
                "X/2": "X/2Pulse",
                "X": "XPulse",
                "-X/2": "-X/2Pulse",
                "Y/2": "Y/2Pulse",
                "Y": "YPulse",
                "-Y/2": "-Y/2Pulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
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
        "constPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "gauss_wf"},
        },
        "XPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "X/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "pi/2_wf", "Q": "zero_wf"},
        },
        "-X/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "-pi/2_wf", "Q": "zero_wf"},
        },
        "YPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "pi_wf"},
        },
        "Y/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "pi/2_wf"},
        },
        "-Y/2Pulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "zero_wf", "Q": "-pi/2_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_pulse},
        "pi_wf": {"type": "arbitrary", "samples": gauss(0.2, 0, 12, pulse_len)},
        "-pi/2_wf": {"type": "arbitrary", "samples": gauss(-0.1, 0, 12, pulse_len)},
        "pi/2_wf": {"type": "arbitrary", "samples": gauss(0.1, 0, 12, pulse_len)},
        "zero_wf": {"type": "constant", "sample": 0},
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
