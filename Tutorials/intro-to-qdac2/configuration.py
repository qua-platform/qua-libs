readout_len = 1000  # Readout length in ns
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},  # in V
                2: {"offset": +0.0},  # in V
            },
            "digital_outputs": {
                1: {}
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # in V
            },
        },
    },
    "elements": {
        "gate": {
            "singleInput": {"port": ("con1", 1)},
            "sticky": {"analog": True, "duration": 100},
            "operations": {
                "step": "constPulse",
            },
        },
        "qdac_trigger1": {
            "singleInput": {"port": ("con1", 2)},
            'digitalInputs': {
                'ext1': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0,
                },
            },
            "operations": {
                "trig": "constPulse",
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
        "constPulse": {
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
        "const_wf": {"type": "constant", "sample": 0.2},  # in V
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
