import numpy as np

######################
# AUXILIARY FUNCTIONS:
######################


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


################
# CONFIGURATION:
################

long_readout_len = 3600
readout_len = 400

qubit_IF = 50e6
rr_IF = 50e6

qubit_LO = 6.345e9
rr_LO = 4.755e9

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # RR I
                2: {"offset": 0.0},  # RR Q
            },
            "digital_outputs": {},
            "analog_inputs": {1: {"offset": 0.0}},
        }
    },
    "elements": {
        "rr": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": rr_LO,
                "mixer": "mixer_RR",
            },
            "intermediate_frequency": rr_IF,
            "operations": {
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,
            "waveforms": {"I": "long_readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "long_integW1": "long_integW1",
                "long_integW2": "long_integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.4},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "long_readout_wf": {"type": "constant", "sample": 0.32},
        "readout_wf": {"type": "constant", "sample": 0.34},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "long_integW1": {
            "cosine": [1.0] * int(long_readout_len / 4),
            "sine": [0.0] * int(long_readout_len / 4),
        },
        "long_integW2": {
            "cosine": [0.0] * int(long_readout_len / 4),
            "sine": [1.0] * int(long_readout_len / 4),
        },
    },
    "mixers": {
        "mixer_RR": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": IQ_imbalance(0.0, 0.0),
            }
        ],
    },
}
