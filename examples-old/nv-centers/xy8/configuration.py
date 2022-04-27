LO_freq = 2.87e9
NV_IF = 0e6  # Makes viewing the pulses simple, can be changed.

pi_length = 80
pi_amplitude = 0.3
pi_amplitude2 = pi_amplitude / ((pi_length - 2) / pi_length)

apd_laser_pulse_len = 2000

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {i + 1: {"offset": 0.0} for i in range(2)},
            "analog_inputs": {
                1: {"offset": 0.0},
            },
        },
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_freq,
            },
            "digitalInputs": {
                "laser": {"buffer": 0, "delay": 0, "port": ("con1", 1)},
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": NV_IF,
            "operations": {
                "pi": "pi",
                "pi_plus_2ns_wait": "pi_2ns",
                "pi_plus_4ns_wait": "pi_4ns",
                "pi_plus_6ns_wait": "pi_6ns",
                "pi_half": "pi_half",
                "readout": "laser_on",
                "laser": "laser_on",
            },
        },
    },
    "pulses": {
        "pi": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_2ns": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {"I": "pi_wf_2ns", "Q": "zero_wf"},
        },
        "pi_4ns": {
            "operation": "control",
            "length": pi_length + 4,
            "waveforms": {"I": "pi_wf_4ns", "Q": "zero_wf"},
        },
        "pi_6ns": {
            "operation": "control",
            "length": pi_length + 4,
            "waveforms": {"I": "pi_wf_6ns", "Q": "zero_wf"},
        },
        "pi_half": {
            "operation": "control",
            "length": pi_length / 2,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "laser_on": {
            "digital_marker": "ON",
            "length": apd_laser_pulse_len,
            "operation": "measurement",
            "waveforms": {
                "I": "zero_wf",
                "Q": "zero_wf",
            },
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "waveforms": {
        "pi_wf": {"type": "constant", "sample": pi_amplitude},
        "pi_wf_2ns": {
            "type": "arbitrary",
            "samples": [0] + [pi_amplitude2] * int(pi_length - 2) + [0],
        },
        "pi_wf_4ns": {
            "type": "arbitrary",
            "samples": [0] * 2 + [pi_amplitude] * int(pi_length) + [0] * 2,
        },
        "pi_wf_6ns": {
            "type": "arbitrary",
            "samples": [0] * 3 + [pi_amplitude2] * int(pi_length - 2) + [0] * 3,
        },
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
}
