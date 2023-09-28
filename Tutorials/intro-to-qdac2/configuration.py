readout_len = 10_000  # Readout length in ns
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},  # in V
                2: {"offset": +0.0},  # in V
            },
            "digital_outputs": {1: {}, 2: {}},
            "analog_inputs": {
                1: {"offset": +0.0},  # in V
            },
        },
    },
    "elements": {
        "qdac_trigger1": {
            "singleInput": {"port": ("con1", 2)},  # Not used
            "digitalInputs": {
                "ext1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "trig": "const_pulse",
            },
        },
        "qdac_trigger2": {
            "singleInput": {"port": ("con1", 2)},  # Not used
            "digitalInputs": {
                "ext2": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "trig": "const_pulse",
            },
        },
        "readout_element": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": 200,
            "smearing": 0,
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "const": "const_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.0},  # in V
        "zero_wf": {"type": "constant", "sample": 0.0},  # in V
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "const_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
    },
}
