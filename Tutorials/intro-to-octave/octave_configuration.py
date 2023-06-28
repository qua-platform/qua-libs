"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
"""
from set_octave import ElementsSettings, octave_settings
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *

# Configure the Octave parameters for each element
qe1 = ElementsSettings("qe1", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal")
qe2 = ElementsSettings("qe2", gain=-10, rf_in_port=["octave1", 2], down_convert_LO_source="Dmd2LO")
# Add the "octave" elements
elements_settings = [qe1, qe2]

###################
# Octave settings #
###################
# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config, log_level="ERROR")
octave_settings(
    qmm=qmm,
    config=config,
    octaves=octaves,
    elements_settings=elements_settings,
    calibration=True,
)
qmm.close()
