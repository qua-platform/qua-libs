"""
Created on 11/11/2021
@author barnaby
"""

import numpy as np
from .pulses import pulses


def get_length(pulse):
    return int(pulses.get(pulse).get("length"))


waveforms = {
    "jump": {"type": "constant", "sample": 0.499},
    "ramp": {"type": "arbitrary", "samples": np.linspace(0, 0.499, get_length("ramp"))},
    "measure": {"type": "constant", "sample": 0.001},
}
