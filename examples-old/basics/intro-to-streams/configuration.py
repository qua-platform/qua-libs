import numpy as np

pulse_len = 1000

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 0e6,
            "operations": {
                "readoutOp": "readoutPulse",
                "readoutOp2": "readoutPulse2",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
    },
    "pulses": {
        "readoutPulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"single": "ramp_wf"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights", "y": "yWeights"},
        },
        "readoutPulse2": {
            "operation": "measure",
            "length": 2 * pulse_len,
            "waveforms": {"single": "ramp_wf2"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights2", "y": "yWeights"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "ramp_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0, -0.5, pulse_len).tolist(),
        },
        "ramp_wf2": {
            "type": "arbitrary",
            "samples": np.linspace(0, -0.5, pulse_len).tolist() + np.linspace(0, -0.5, pulse_len).tolist(),
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "xWeights": {
            "cosine": [1.0] * (pulse_len // 4),
            "sine": [0.0] * (pulse_len // 4),
        },
        "xWeights2": {
            "cosine": [1.0] * (2 * pulse_len // 4),
            "sine": [0.0] * (2 * pulse_len // 4),
        },
        "yWeights": {
            "cosine": [0.0] * (pulse_len // 4),
            "sine": [1.0] * (pulse_len // 4),
        },
    },
}
