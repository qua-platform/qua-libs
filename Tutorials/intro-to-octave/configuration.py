from set_octave import OctaveUnit, octave_declaration

######################
# Network parameters #
######################
qop_ip = "1127.0.0.1"
qop_port = 80

############################
# Set octave configuration #
############################
octave_1 = OctaveUnit("octave1", qop_ip, port=50, con="con1", clock="Internal")
# octave_2 = OctaveUnit("octave2", opx_ip, port=51, con="con1", clock="Internal")
# Custom port mapping example
port_mapping = [
    {
        ("con1", 1): ("octave1", "I1"),
        ("con1", 2): ("octave1", "Q1"),
        ("con1", 3): ("octave1", "I2"),
        ("con1", 4): ("octave1", "Q2"),
        ("con1", 5): ("octave1", "I3"),
        ("con1", 6): ("octave1", "Q3"),
        ("con1", 7): ("octave1", "I4"),
        ("con1", 8): ("octave1", "Q4"),
        ("con1", 9): ("octave1", "I5"),
        ("con1", 10): ("octave1", "Q5"),
    }
]

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
# OPX configuration #
#####################
IF = 50e6
LO = 6e9

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
                3: {"offset": 0.0},
                4: {"offset": 0.0},
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO,
                "mixer": "octave_octave1_1",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 1),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe2": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": LO,
                "mixer": "octave_octave1_2",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 2),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe3": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": LO,
                "mixer": "octave_octave1_3",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 3),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe4": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": LO,
                "mixer": "octave_octave1_4",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 4),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe5": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": LO,
                "mixer": "octave_octave1_5",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 5),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
    },
    "pulses": {
        "const": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.125,
        },
        "readout_wf": {
            "type": "constant",
            "sample": 0.125,
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "OFF": {"samples": [(0, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
    },
    "mixers": {
        "octave_octave1_1": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        "octave_octave1_2": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        "octave_octave1_3": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        "octave_octave1_4": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        "octave_octave1_5": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}
