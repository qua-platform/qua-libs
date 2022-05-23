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
        "qubit": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0e6,
            "operations": {
                "cw1": "const_pulse1",
                "cw2": "const_pulse2",
            },
        },
        "resonator": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0e6,
            "operations": {
                "cw2": "const_pulse2",
            },
        },
    },
    "pulses": {
        "const_pulse1": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf1"},
        },
        "const_pulse2": {
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
