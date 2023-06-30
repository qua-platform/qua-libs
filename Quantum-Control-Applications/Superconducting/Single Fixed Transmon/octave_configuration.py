"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
You need to run this file in order to update the Octaves with the new parameters.
"""
from set_octave import ElementsSettings, octave_settings
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_with_octave import *

# Configure the Octave parameters for each element
resonator = ElementsSettings("resonator", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal")
qubit = ElementsSettings("qubit", gain=-10, rf_in_port=["octave1", 2])
# Add the "octave" elements
elements_settings = [resonator, qubit]

###################
# Octave settings #
###################
# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config, log_level="ERROR")
octave_settings(
    qmm=qmm,
    config=config,
    octaves=octaves,
    elements_settings=elements_settings,
    calibration=True,
)
qmm.close()
