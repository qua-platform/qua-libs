import numpy as np

RR_lo_freq = 3e9
IF_freq = 0

opx_one = "con1"
config = {
    "version": 1,
    "controllers": {
        opx_one: {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0},
                2: {"offset": 0},
                3: {"offset": 0},
                4: {"offset": 0},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": (opx_one, 1)},
            "intermediate_frequency": IF_freq,
            "operations": {"measurement": "photon1_pulse", "short": "short_pulse"},
            "outputs": {"out1": (opx_one, 1)},
            "time_of_flight": 28,
            "smearing": 0,
        },
        "qe2": {
            "singleInput": {"port": (opx_one, 2)},
            "intermediate_frequency": IF_freq,
            "operations": {"measurement": "photon2_pulse", "short": "short_pulse"},
            "outputs": {"out1": (opx_one, 2)},
            "time_of_flight": 28,
            "smearing": 0,
        },
        "qeDig": {
            "digitalInputs": {"trig_cam": {"buffer": 0, "delay": 144, "port": (opx_one, 9)}},
            "operations": {"marker": "marker_in"},
        },
        "qeCol": {
            "singleInputCollection": {"inputs": {"o1": (opx_one, 3), "o2": (opx_one, 4)}},
            "intermediate_frequency": IF_freq,
            "operations": {"measurement": "readout_pulse"},
            "outputs": {"out1": (opx_one, 1)},
            "time_of_flight": 100,
            "smearing": 0,
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": 1e5,
            "waveforms": {
                # 'I': 'const_wf',
                "single": "const_wf"
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
        "short_pulse": {
            "operation": "measurement",
            "length": 16,
            "waveforms": {
                # 'I': 'const_wf',
                "single": "const_wf"
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
        "marker_in": {
            "digital_marker": "trig_wf0",
            "length": 100,
            "operation": "control",
        },
        "photon1_pulse": {
            "operation": "measurement",
            "length": 200,
            "waveforms": {"single": "photon_1"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
        "photon2_pulse": {
            "operation": "measurement",
            "length": 200,
            "waveforms": {"single": "photon_2"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": 0.4},
        "photon_1": {"type": "arbitrary", "samples": [0, 0, 0, 0]},
        "photon_2": {"type": "arbitrary", "samples": [0, 0, 0, 0]},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "trig_wf0": {"samples": [(1, 100), (0, 0)]},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * 100,
            "sine": [0.0] * 100,
        },
        "integW2": {
            "cosine": [0.0] * 100,
            "sine": [1.0] * 100,
        },
    },
    "mixers": {
        "mixer_RR": [
            {
                "intermediate_frequency": IF_freq,
                "lo_frequency": RR_lo_freq,
                "correction": [1, 0, 0, 1],
            }
        ],
    },
}
