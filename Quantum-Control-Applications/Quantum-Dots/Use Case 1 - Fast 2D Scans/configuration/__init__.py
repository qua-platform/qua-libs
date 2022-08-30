"""
Created on 11/11/2021
@author barnaby
"""

from .controllers import controllers
from .elements import elements
from .pulses import pulses
from .waveforms import waveforms
from .integration_weights import integration_weights
from .digital_waveforms import digital_waveforms
from qualang_tools.units import unit

qop_ip = "127.0.0.1"
u = unit()

config = {
    "version": 1,
    "controllers": controllers,
    "elements": elements,
    "pulses": pulses,
    "waveforms": waveforms,
    "digital_waveforms": digital_waveforms,
    "integration_weights": integration_weights,
}
