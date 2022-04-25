pulse_len = 1e3

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 5e6,
            "operations": {
                "const": "constPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": pulse_len,  # in ns
            "waveforms": {"single": "const_wf"},
        }
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
}
