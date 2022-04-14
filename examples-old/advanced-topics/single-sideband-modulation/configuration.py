config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0},
                2: {"offset": +0},
            },
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "ssbElement": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2)},
            "digitalInputs": {
                "switch1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 50e6,
            "operations": {"ssb": "ssbPulse"},
        },
    },
    "pulses": {
        "ssbPulse": {
            "operation": "control",
            "length": 1,
            "waveforms": {"I": "wf1", "Q": "wf2"},
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "wf1": {"type": "arbitrary", "samples": [0], "sampling_rate": 2e6},
        "wf2": {"type": "arbitrary", "samples": [0], "sampling_rate": 2e6},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
}
