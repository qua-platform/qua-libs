config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "digital_outputs": {1: {}},
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 5e6,
            "operations": {
                "const": "constPulse",
            },
        },
        "qe2": {
            "singleInput": {"port": ("con1", 1)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 200,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 5e6,
            "operations": {
                "const": "constPulse",
                "const_trig": "constPulse_trig",
                "const_stutter": "constPulse_stutter",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "constPulse_trig": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "trig",
        },
        "constPulse_stutter": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "stutter",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "trig": {"samples": [(1, 100)]},
        "stutter": {"samples": [(1, 100), (0, 200), (1, 76), (0, 10), (1, 0)]},
    },
}
