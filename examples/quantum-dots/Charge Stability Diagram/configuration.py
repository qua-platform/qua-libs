readout_len = 1000
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},

            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
            "digital_outputs": {1: {}, 2: {}},
        }
    },
    "elements": {
        "PG1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "PG2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "QPC": {
            "singleInput": {"port": ("con1", 3)},
            "outputs": {"out1": ("con1", 1)},
            "intermediate_frequency": 0,
            "operations": {
                "readout": "zeroPulse",
            },
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": readout_len,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "zeroPulse": {
            "operation": "measurement",
            "length": readout_len,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {"integW": "integW"}
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 1},
        "zero_wf": {"type": "constant", "sample": 0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW": {
            "cosine": [1] * (readout_len // 4),
            "sine": [1] * (readout_len // 4)
        }
    }
}
