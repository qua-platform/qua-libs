import numpy as np

gauss_len = 4


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


readout_len = 400
qubit_IF = 100e6
manual_ssb_IF = 100e6
use_manual_ssb = False

rr_IF = 0
qubit_LO = 5117.22e6
rr_LO = 5117.22e6

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
                "X/2": "DRAG_PULSE_X/2",
                "-X/2": "DRAG_PULSE_-X/2",
                "Y/2": "DRAG_PULSE_Y/2",
                "-Y/2": "DRAG_PULSE_-Y/2",
                "X": "DRAG_PULSE_X",
                "Y": "DRAG_PULSE_Y",
                "random_clifford_seq": "random_sequence",
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
        "DRAG_PULSE_X/2": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_X/2", "Q": "DRAG_gauss_der_wf_X/2"},
        },
        "DRAG_PULSE_Y/2": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_Y/2", "Q": "DRAG_gauss_der_wf_Y/2"},
        },
        "DRAG_PULSE_-X/2": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_-X/2", "Q": "DRAG_gauss_der_wf_-X/2"},
        },
        "DRAG_PULSE_-Y/2": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_-Y/2", "Q": "DRAG_gauss_der_wf_-Y/2"},
        },
        "DRAG_PULSE_X": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_X", "Q": "DRAG_gauss_der_wf_X"},
        },
        "DRAG_PULSE_Y": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "DRAG_gauss_wf_Y", "Q": "DRAG_gauss_der_wf_Y"},
        },
        "random_sequence": {
            "operation": "control",
            "length": None,
            "waveforms": {"I": "random_I", "Q": "random_Q"},
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
        "zero_wf": {"type": "constant", "sample": 0.0},
        "DRAG_gauss_wf_X": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_X": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_wf_Y": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_Y": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_wf_X/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_X/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_wf_-X/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_-X/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_wf_Y/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_Y/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_wf_-Y/2": {"type": "arbitrary", "samples": []},
        "DRAG_gauss_der_wf_-Y/2": {"type": "arbitrary", "samples": []},
        "readout_wf": {"type": "arbitrary", "samples": []},
        "random_I": {
            "type": "arbitrary",
            "samples": [],
        },
        "random_Q": {
            "type": "arbitrary",
            "samples": [],
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * readout_len,
            "sine": [0.0] * readout_len,
        },
        "integW2": {
            "cosine": [0.0] * readout_len,
            "sine": [1.0] * readout_len,
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
