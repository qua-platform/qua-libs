import numpy as np

readout_len = 1000

config = {
    "version": 1,
    "controllers": {
        "con1": {
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
            "intermediate_frequency": 100e6,
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measure",
            "length": readout_len,
            "waveforms": {"single": "ramp_wf"},
            "digital_marker": "ON",
            "integration_weights": {"cos": "cosine_weights", "sin": "sine_weights"},
        },
        "long_readout_pulse": {
            "operation": "measure",
            "length": 2 * readout_len,
            "waveforms": {"single": "ramp_wf2"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "long_cosine_weights",
                "sin": "sine_weights",
            },
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "ramp_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0, -0.5, readout_len).tolist(),
        },
        "ramp_wf2": {
            "type": "arbitrary",
            "samples": np.linspace(0, -0.5, readout_len).tolist() + np.linspace(0, -0.5, readout_len).tolist(),
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "long_cosine_weights": {
            "cosine": [(1.0, 2 * readout_len)],
            "sine": [(0.0, 2 * readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
    },
}
