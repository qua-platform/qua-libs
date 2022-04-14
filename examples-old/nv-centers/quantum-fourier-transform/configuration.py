NV_IF = 50e6
N_IF = 5e6
C414_IF = 5e6
C90_IF = 3e6

N_hyperfine = 2.16e6
C414_hyperfine = 414e3
C90_hyperfine = 89e3

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
        "NV": {
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
                "pi": "pi",
                "pi_N+": "CNOT_N+",
                "pi_N0": "CNOT_N0",
                "pi_N-": "CNOT_N0",
                "pi_C414+": "CNOT_C414+",
                "pi_C414-": "CNOT_C414-",
                "pi_C90+": "CNOT_C90+",
                "pi_C90-": "CNOT_C90-",
                "pi_over_two_N+": "pi_over_two_N+",
                "pi_over_two_N0": "pi_over_two_N0",
                "pi_over_two_N-": "pi_over_two_N0",
                "pi_over_two_C414+": "pi_over_two_C414+",
                "pi_over_two_C414-": "pi_over_two_C414-",
                "pi_over_two_C90+": "pi_over_two_C90+",
                "pi_over_two_C90-": "pi_over_two_C90-",
                "pi_over_two_C90_C414": "pi_over_two_C90_C414",
                "pi_over_six_C90_N1": "pi_over_six_C90_N1",
                "two_pi_over_six_C90_N2": "two_pi_over_six_C90_N2",
                "pi_over_three_C414_N1": "pi_over_three_C414_N1",
                "two_pi_over_three_C414_N2": "two_pi_over_three_C414_N2",
                "readout": "laser_on",
                "laser": "laser_on",
            },
        },
        "N": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": N_IF,
            "operations": {
                "pi_1_-1": "pi_1_-1_N",
                "pi_1_0": "pi_1_0_N",
                "pi_0_-1": "pi_0_-1_N",
                "pi_0_1": "pi_1_0_N",
                "pi_-1_0": "pi_0_-1_N",
                "pi_-1_1": "pi_1_-1_N",
                "Chrestenson": "Chrestenson_N",
                "pi_-1_-1": "fake_pulse",
                "pi_0_0": "fake_pulse",
                "pi_1_1": "fake_pulse",
            },
        },
        "C414": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": C414_IF,
            "operations": {
                "pi": "pi_C414",
                "Hadamard": "Hadmard_C414",
            },
        },
        "C90": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "time_of_flight": 28,
            "smearing": 0,
            "intermediate_frequency": C90_IF,
            "operations": {
                "pi": "pi_C90",
                "Hadamard": "Hadmard_C90",
            },
        },
    },
    "pulses": {
        "pi": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_N+": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_N0": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_N0": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_C414+": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_C414-": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_C90+": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "CNOT_C90-": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "pi_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_N+": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_N0": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_N0": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_C414+": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_C414-": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_C90+": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_C90-": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "pi_over_two_NV_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_two_C90_C414": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_over_two_C90_C414_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_six_C90_N1": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "pi_over_six_C90_N1_wf",
                "Q": "zero_wf",
            },
        },
        "two_pi_over_six_C90_N2": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "two_pi_over_six_C90_N2_wf",
                "Q": "zero_wf",
            },
        },
        "pi_over_three_C414_N1": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "pi_over_three_C414_N1_wf",
                "Q": "zero_wf",
            },
        },
        "two_pi_over_three_C414_N2": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "I": "two_pi_over_three_C414_N2_wf",
                "Q": "zero_wf",
            },
        },
        "fake_pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "zero_wf",
            },
        },
        "pi_1_-1_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "pi_1_0_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "pi_0_-1_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "pi_1_0_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "pi_0_-1_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "pi_1_-1_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "pi_N_wf",
            },
        },
        "Chrestenson_N": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "Chrestenson_N_wf",
            },
        },
        "Hadmard_C414": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "single": "Hadmard_C414_wf",
            },
        },
        "Hadmard_C90": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "single": "Hadmard_C90_wf",
            },
        },
        "pi_C414": {
            "operation": "control",
            "length": 5000,
            "waveforms": {
                "single": "pi_C414_wf",
            },
        },
        "pi_C90": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "single": "pi_C90_wf",
            },
        },
        "laser_on": {
            "digital_marker": "ON",
            "length": 2000,
            "operation": "measurement",
            "waveforms": {
                "I": "zero_wf",
                "Q": "zero_wf",
            },
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "waveforms": {
        "pi_NV_wf": {"type": "constant", "sample": 0.2},
        "pi_over_two_NV_wf": {"type": "constant", "sample": 0.2},
        "pi_over_two_C90_C414_wf": {"type": "constant", "sample": 0.2},
        "pi_over_six_C90_N1_wf": {"type": "constant", "sample": 0.2},
        "two_pi_over_six_C90_N2_wf": {"type": "constant", "sample": 0.2},
        "pi_over_three_C414_N1_wf": {"type": "constant", "sample": 0.2},
        "two_pi_over_three_C414_N2_wf": {"type": "constant", "sample": 0.2},
        "pi_N_wf": {"type": "constant", "sample": 0.2},
        "Chrestenson_N_wf": {"type": "constant", "sample": 0.2},
        "Hadmard_C414_wf": {"type": "constant", "sample": 0.2},
        "Hadmard_C90_wf": {"type": "constant", "sample": 0.2},
        "pi_C414_wf": {"type": "constant", "sample": 0.2},
        "pi_C90_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
}
