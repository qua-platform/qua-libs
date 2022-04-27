import numpy as np

pulse_len = 100
qubit_IF = 50e6
RR_IF = 50e6
RR_LO = 1e9
qubit_LO = 2e9
const_pulse_len = 1000
pulse_len = 1000
readout_len = pulse_len
step_num = 300


def IQ_imbalance_correction(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Iqb
                2: {"offset": +0.0},  # Qqb
                3: {"offset": +0.0},  # Irr
                4: {"offset": +0.0},  # Qrr
                5: {"offset": -0.4},  # flux line
            },
            "analog_inputs": {
                1: {"offset": +0.0},
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
                "test_pulse": "MixedConstPulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
        "RR": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": RR_LO,
                "mixer": "mixer_RR",
            },
            "outputs": {"out1": ("con1", 1)},
            "intermediate_frequency": RR_IF,
            "operations": {
                "measure": "measurePulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
        "flux_line": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": 0,
            "hold_offset": {"duration": 1000},
            "operations": {
                "tune": "constPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "flux_tune_wf"},
        },
        "MixedConstPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "measurePulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "const_wf2": {"type": "constant", "sample": 0.8 / 300},
        "flux_tune_wf": {"type": "constant", "sample": 1 / step_num},
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
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance_correction(0, 0),
            }
        ],
        "mixer_RR": [
            {
                "intermediate_frequency": RR_IF,
                "lo_frequency": RR_LO,
                "correction": IQ_imbalance_correction(0, 0),
            }
        ],
    },
}
