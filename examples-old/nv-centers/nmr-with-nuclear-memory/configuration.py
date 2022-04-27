NV_IF = 50e6
memory_IF = 5e6
sample_IF = 3e6

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {i + 1: {"offset": 0.0} for i in range(4)},
            "analog_inputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
        },
    },
    "elements": {
        "sensor": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": 2.87e9,
            },
            "digitalInputs": {
                "laser": {"buffer": 0, "delay": 0, "port": ("con1", 1)},
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": NV_IF,
            "operations": {
                "pi_x": "pi_x",
                "pi_2_x": "pi_2_x",
                "pi_y": "pi_y",
                "pi_2_y": "pi_2_y",
                "readout": "laser_on",
                "laser": "laser_on",
            },
        },
        "memory": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": memory_IF,
            "operations": {
                "pi_2_x": "pi_2_x_memory",
                "pi_2_y": "pi_2_y_memory",
                "pi_x": "pi_x_memory",
                "pi_y": "pi_y_memory",
            },
        },
        "sample": {
            "singleInput": {"port": ("con1", 4)},
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": sample_IF,
            "operations": {"pi": "pi_sample", "pi_2": "pi_2_sample"},
        },
    },
    "pulses": {
        "pi_x": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "wf1",
                "Q": "wNV_IF",
            },
        },
        "pi_y": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "wNV_IF",
                "Q": "wf1",
            },
        },
        "pi_2_x": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "wf1",
                "Q": "wNV_IF",
            },
        },
        "pi_2_y": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "wNV_IF",
                "Q": "wf1",
            },
        },
        "pi_2_x_memory": {
            "operation": "control",
            "length": 4000,
            "waveforms": {
                "single": "wf1",
            },
        },
        "pi_2_y_memory": {
            "operation": "control",
            "length": 4000,
            "waveforms": {
                "single": "wf1",
            },
        },
        "pi_x_memory": {
            "operation": "control",
            "length": 8000,
            "waveforms": {
                "single": "wf1",
            },
        },
        "pi_y_memory": {
            "operation": "control",
            "length": 8000,
            "waveforms": {
                "single": "wf1",
            },
        },
        "pi_sample": {
            "operation": "control",
            "length": 8000,
            "waveforms": {"single": "wf1"},
        },
        "pi_2_sample": {
            "operation": "control",
            "length": 4000,
            "waveforms": {"single": "wf1"},
        },
        "single": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "wf1",
            },
        },
        "laser_on": {
            "digital_marker": "ON",
            "length": 2000,
            "operation": "measurement",
            "waveforms": {
                "I": "wNV_IF",
                "Q": "wNV_IF",
            },
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "waveforms": {
        "wf1": {"type": "constant", "sample": 0.45},
        "wNV_IF": {"type": "constant", "sample": 0.0},
    },
}
