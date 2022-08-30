"""
Created on 11/11/2021
@author barnaby
"""

from .pulses import pulses


def get_length(pulse):
    return int(pulses.get(pulse).get("length"))


integration_weights = {
    "cos": {
        "cosine": [(1.0, get_length("measure"))],
        "sine": [(0.0, get_length("measure"))],
    },
    "sin": {
        "cosine": [(0.0, get_length("measure"))],
        "sine": [(1.0, get_length("measure"))],
    },
}
