
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
            },
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0e6,
            "operations": {
                "const1": "constPulse1",
                "const2": "constPulse2",
            },
        },

        "qe2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0e6,
            "operations": {
                "const2": "constPulse2",
            },
        },
    },
    "pulses": {
        "constPulse1": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf1"},
        },
        "constPulse2": {
            "operation": "control",
            "length": 2000,  # in ns
            "waveforms": {"single": "const_wf2"},
        },
    },
    "waveforms": {
        "const_wf1": {"type": "constant", "sample": 0.2},
        "const_wf2": {"type": "constant", "sample": 0.4},
    },
}