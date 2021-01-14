RR_lo_freq = 3e9
IF_freq = 10e6

opx_one = "con1"
config = {
    "version": 1,
    "controllers": {
        opx_one: {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0},
                2: {"offset": 0},
                3: {"offset": 0},
                4: {"offset": 0},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": (opx_one, 1)},
            "intermediate_frequency": IF_freq,
            "operations": {
                "measurement": "readout_pulse",
            },
            "outputs": {"out1": (opx_one, 1)},
            "time_of_flight": 100,
            "smearing": 0,
        },
    },
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": 500,
            "waveforms": {"single": "const_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": 0.4},
    },
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * 100,
            "sine": [0.0] * 100,
        },
        "integW2": {
            "cosine": [0.0] * 100,
            "sine": [1.0] * 100,
        },
    },
    "mixers": {
        "mixer_RR": [
            {
                "intermediate_frequency": IF_freq,
                "lo_frequency": RR_lo_freq,
                "correction": [1, 0, 0, 1],
            }
        ],
    },
}
