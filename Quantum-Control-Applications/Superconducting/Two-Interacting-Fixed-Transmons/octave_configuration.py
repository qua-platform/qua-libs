"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
You need to run this file in order to update the Octaves with the new parameters.
"""
from set_octave import ElementsSettings, octave_settings
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_with_octave import *

# Configure the Octave parameters for each element
rr = ElementsSettings("rr1", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal")
q1_xy = ElementsSettings("q1_xy", gain=-10)
q2_xy = ElementsSettings("q2_xy", gain=-10)
# Add the "octave" elements
elements_settings = [rr, q1_xy, q2_xy]

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
