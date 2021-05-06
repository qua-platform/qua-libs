config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # dummy
                2: {"offset": +0.0},  # a-espin I
                3: {"offset": +0.0},  # a-espin Q
                4: {"offset": +0.0},  # b-espin I
                5: {"offset": +0.0},  # b-espin Q
                6: {"offset": +0.0},  # n-spin manip
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
                9: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "a-reset": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "a-ro": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "time_of_flight": 180,
            "smearing": 0,
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "a-init": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "b-init": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 6),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "b-reset": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "b-ro": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 2)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 5),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "time_of_flight": 180,
            "smearing": 0,
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
            },
        },
        "a-espin": {
            "mixInputs": {"I": ("con1", 2), "Q": ("con1", 3)},
            "intermediate_frequency": 50e6,
            "operations": {"CNOT": "CNOT", "PI": "PI", "PI2": "PI2"},
        },
        "b-espin": {
            "mixInputs": {"I": ("con1", 4), "Q": ("con1", 5)},
            "intermediate_frequency": 50e6,
            "operations": {"CNOT": "CNOT", "PI": "PI", "PI2": "PI2"},
        },
        "nspin": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 50e6,
            "operations": {
                "playOp": "constPulse",
            },
        },
    },
    "pulses": {
        "zeroPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
        },
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "CNOT": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "PI": {
            "operation": "control",
            "length": 120,  # in ns
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "PI2": {
            "operation": "control",
            "length": 52,  # in ns
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "readoutPulse": {
            "operation": "measure",
            "length": 52,
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights", "y": "yWeights"},
        },
        "short_readoutPulse": {
            "operation": "measure",
            "length": 16,
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights", "y": "yWeights"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "xWeights": {
            "cosine": [1.0] * 52,
            "sine": [0.0] * 52,
        },
        "yWeights": {
            "cosine": [0.0] * 52,
            "sine": [1.0] * 52,
        },
    },
}
