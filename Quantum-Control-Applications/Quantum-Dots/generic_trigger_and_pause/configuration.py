"""
Created on 09/10/2022
@author Jonathan R.
"""
from qualang_tools.units import unit

#############
# VARIABLES #
#############

u = unit()
qop_ip = "172.16.2.102"
readout_time = 256  # ns

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {1: {"offset": 0.0},  # G1
                               2: {"offset": 0.0},  # G2
                               3: {"offset": 0.0},  # I qubit
                               4: {"offset": 0.0},  # Q qubit
                               5: {"offset": 0.0},  # I resonator
                               6: {"offset": 0.0},  # Q resonator
                               7: {'offset': 0.0},   # readout element bias (compensation)
                               8: {'offset': 0.0},   # readout element bias (compensation)
                               9: {'offset': 0.0},   # readout element bias (compensation)
                               10: {'offset': 0.0},   # readout element bias (compensation)
                               },
            "digital_outputs":
                {i: {} for i in range(1, 11)},

            "analog_inputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
        },
    },
    "elements": {
        "G1_sticky": {
            "singleInput": {"port": ("con1", 1)},
            "hold_offset": {"duration": 100},
            "operations": {
                "constant": "constant",
            },
        },
        "G2_sticky": {
            "singleInput": {"port": ("con1", 2)},
            "hold_offset": {"duration": 100},
            "operations": {
                "constant": "constant",
            },
        },
        "G1": {
            "singleInput": {"port": ("con1", 1)},
            "operations": {
                "constant": "constant",
            },
        },
        "G2": {
            "singleInput": {"port": ("con1", 2)},
            "operations": {
                "constant": "constant",
            },
        },
        "RF": {
            "singleInput": {"port": ("con1", 5)},
            "time_of_flight": 200,
            "smearing": 0,
            "intermediate_frequency": 100e6,
            "outputs": {"out1": ("con1", 1)},
            "operations": {"measure": "measure"},
        },
        'trigger_x': {
            "digitalInputs": {
                    "trigger_qdac": {
                        'port': ('con1', 8),
                        'delay': 0,
                        'buffer': 0
                    }
                },
            'operations': {
                'trig': 'trigger'
            }
        },
        'trigger_y': {
            "digitalInputs": {
                    "trigger_qdac": {
                        'port': ('con1', 10),
                        'delay': 0,
                        'buffer': 0
                    }
                },
            'operations': {
                'trig': 'trigger'
            }
        },

    },
    "pulses": {
        "constant": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "constant",
            },
        },
        "trigger": {
            "operation": "control",
            "length": 100,
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
    }
    ,
    "waveforms": {
        "constant": {"type": "constant", "sample": 0.5},
        "measure": {"type": "constant", "sample": 0.02},
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
