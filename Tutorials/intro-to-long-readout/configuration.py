readout_len = 1000
fem = 5
port = 7

config = {
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                fem: {
                    "type": "LF",
                    "analog_outputs": {
                        port: {
                            "offset": 0.0,
                            "output_mode": "direct",
                            "sampling_rate": 1e9,
                            "upsampling_mode": "mw",
                        },
                    },
                    "digital_outputs": {},
                    "analog_inputs": {
                        1: {"offset": 0.0, "gain_db": 0, "sampling_rate": 1e9},
                    },
                }
            },
        }
    },
    "elements": {
        "resonator": {
            "singleInput": {
                "port": ("con1", fem, port),
            },
            "intermediate_frequency": 10e6,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", fem, 1),
            },
            "time_of_flight": 28,
            "smearing": 0,
        },
        "resonator_twin": {
            "singleInput": {
                "port": ("con1", fem, port),
            },
            "intermediate_frequency": 10e6,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", fem, 1),
            },
            "time_of_flight": 300,
            "smearing": 0,
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_pulse_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "readout_pulse_wf": {"type": "constant", "sample": 0.2},
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
    "mixers": {},
}
