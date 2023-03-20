"""
Created on 09/10/2022
@author Jonathan R.
"""
from qualang_tools.units import unit

#############
# VARIABLES #
#############

u = unit()
qop_ip = "172.16.2.103"
readout_time = 512  # ns
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # G1
                2: {"offset": 0.0},  # G2
                3: {"offset": 0.0},  # I qubit
                4: {"offset": 0.0},  # Q qubit
                5: {"offset": 0.0},  # I resonator
                6: {"offset": 0.0},  # Q resonator
            },
            "digital_outputs": {
                1: {},  # trigger for external device
            },
            "analog_inputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
        },
    },
    "elements": {
        "G1_sticky": {
            "singleInput": {"port": ("con1", 1)},
            "hold_offset": {"duration": 200},
            "operations": {
                "sweep": "sweep",
            },
        },
        "G2_sticky": {
            "singleInput": {"port": ("con1", 2)},
            "hold_offset": {"duration": 200},
            "operations": {
                "sweep": "sweep",
            },
        },
        "G1": {
            "singleInput": {"port": ("con1", 1)},
            "operations": {
                "sweep": "sweep",
            },
        },
        "G2": {
            "singleInput": {"port": ("con1", 2)},
            "operations": {
                "sweep": "sweep",
            },
        },
        "RF": {
            "singleInput": {"port": ("con1", 3)},
            "time_of_flight": 200,
            "smearing": 0,
            "intermediate_frequency": 100e6,
            "outputs": {"out1": ("con1", 1)},
            "operations": {"measure": "measure"},
        },
        "QDAC": {
            "singleInput": {"port": ("con1", 3)},
            "operations": {"trigger": "trigger"},
        },
    },
    "pulses": {
        "sweep": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "sweep",
            },
        },
        "trigger": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "zero"},
            "digital_marker": "ON",
        },
        "measure": {
            "operation": "measurement",
            "length": readout_time,
            "waveforms": {"single": "measure"},
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cos",
                "sin": "sin",
            },
        },
    },
    "waveforms": {
        "sweep": {"type": "constant", "sample": 0.499},
        "measure": {"type": "constant", "sample": 0.001},
        "zero": {"type": "constant", "sample": 0.00},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "cos": {
            "cosine": [(1.0, readout_time)],
            "sine": [(0.0, readout_time)],
        },
        "sin": {
            "cosine": [(0.0, readout_time)],
            "sine": [(1.0, readout_time)],
        },
    },
}
