"""
Created on 11/11/2021
@author barnaby
"""

pulses = {
    "ramp": {"operation": "control", "length": 100, "waveforms": {"single": "ramp"}},
    "jump": {
        "operation": "control",
        "length": 100,
        "waveforms": {"single": "jump"},
    },
    "measure": {
        "operation": "measurement",
        "length": 512,
        "waveforms": {"single": "measure"},
        "digital_marker": "ON",
        "integration_weights": {
            "cos": "cos",
            "sin": "sin",
        },
    },
}
