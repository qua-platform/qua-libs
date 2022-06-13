import numpy as np
from scipy.signal.windows import gaussian

################
# CONFIGURATION:
################

qubit_IF = 50.123e6
reflectometry_IF = 300e6

lockin_freq = 80  # Hz
# readout_pulse_length = round(1 / lockin_freq * 1e9)  # ns
readout_repeats = 20  # To speed up compilation
readout_pulse_length = round(1 / lockin_freq * 1e9 / readout_repeats)  # ns
cw_readout_pulse_length = int(100e3)

cw_amp = 0.3

reflect_amp = 0.03

block_amp = 0.3
block_length = 20

pi_amp = 0.3
pi_half_amp = 0.15
pi_length = 40  # ns
pi_half_length = 40  # ns

gaussian_amp = 0.3
gaussian_length = 20

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # Coulomb blockage
                2: {"offset": 0.0},  # Qubit I
                3: {"offset": 0.0},  # Qubit Q
                4: {"offset": 0.0},  # RF SET
            },
            "digital_outputs": {
                1: {},  # To indicate measure time
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 6},  # Transport readout
            },
        }
    },
    "elements": {
        "CB": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "operations": {
                "block": "block_pulse",
            },
        },
        "qubit": {
            "mixInputs": {
                "I": ("con1", 2),
                "Q": ("con1", 3),
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_pulse",
                "gauss": "gaussian_pulse",
                "pi_half": "pi_half_pulse",
            },
        },
        "readout": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "intermediate_frequency": reflectometry_IF,
            "operations": {
                "readout": "readout_pulse",
                "reflectometry": "reflectometry_readout_pulse",
                "cw_reflectometry": "cw_reflectometry_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 200,
            "smearing": 0,
            "digitalInputs": {
                "indicator": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
        },
    },
    "pulses": {
        "block_pulse": {
            "operation": "control",
            "length": block_length,
            "waveforms": {
                "single": "block_wf",
            },
        },
        "cw_pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_length,
            "waveforms": {
                "I": "gaussian_wf",
                "Q": "zero_wf",
            },
        },
        "pi_pulse": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_half_length,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_length,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "const": "cosine_weights",
            },
            "digital_marker": "ON",
        },
        "reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
        "cw_reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": cw_readout_pulse_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cw_cosine_weights",
                "sin": "cw_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": cw_amp},
        "reflect_wf": {"type": "constant", "sample": reflect_amp},
        "block_wf": {"type": "constant", "sample": block_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gaussian_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in gaussian_amp * gaussian(gaussian_length, gaussian_length / 5)],
        },
        "pi_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_length, pi_length / 5)]},
        "pi_half_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_half_amp * gaussian(pi_half_length, pi_half_length / 5)],
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_pulse_length)],
            "sine": [(0.0, readout_pulse_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_pulse_length)],
            "sine": [(1.0, readout_pulse_length)],
        },
        "cw_cosine_weights": {
            "cosine": [(1.0, cw_readout_pulse_length)],
            "sine": [(0.0, cw_readout_pulse_length)],
        },
        "cw_sine_weights": {
            "cosine": [(0.0, cw_readout_pulse_length)],
            "sine": [(1.0, cw_readout_pulse_length)],
        },
    },
}
