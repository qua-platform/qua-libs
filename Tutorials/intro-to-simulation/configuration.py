readout_len = 200
const_len = 200
const_amp = 0.2

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
                3: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        },
        "con2": {
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 100e6,
            "operations": {
                "const": "const_pulse",
                "readout": "readout_pulse",
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe2": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 50e6,
            "operations": {
                "const": "const_pulse",
            },
        },
        "qe3": {
            "singleInput": {"port": ("con2", 1)},
            "intermediate_frequency": 50e6,
            "operations": {
                "const": "const_pulse",
            },
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
            "integration_weights": {"cos": "cosine_weights", "sin": "sine_weights"},
        },
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
    },
}
