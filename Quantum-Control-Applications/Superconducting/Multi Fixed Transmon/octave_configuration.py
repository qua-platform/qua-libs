"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
"""
from set_octave import ElementsSettings, octave_settings
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from quam import QuAM

# Configure the Octave parameters for each element
rr0 = ElementsSettings("rr0", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr1 = ElementsSettings("rr1", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr2 = ElementsSettings("rr2", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr3 = ElementsSettings("rr3", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr4 = ElementsSettings("rr4", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr5 = ElementsSettings("rr5", gain=0, rf_in_port=["octave1", 1], down_convert_LO_source="Internal") 
rr6 = ElementsSettings("rr6", gain=0, rf_in_port=["octave1", 2], down_convert_LO_source="Dmd2LO") 
rr7 = ElementsSettings("rr7", gain=0, rf_in_port=["octave1", 2], down_convert_LO_source="Dmd2LO") 
rr8 = ElementsSettings("rr8", gain=0, rf_in_port=["octave1", 2], down_convert_LO_source="Dmd2LO") 

q0 = ElementsSettings("q0", gain=0)
q1 = ElementsSettings("q1", gain=0)
q2 = ElementsSettings("q2", gain=0)
q3 = ElementsSettings("q3", gain=0)
q4 = ElementsSettings("q3", gain=0)
q5 = ElementsSettings("q3", gain=0)
q6 = ElementsSettings("q3", gain=0)
q7 = ElementsSettings("q3", gain=0)
q8 = ElementsSettings("q3", gain=0)
# Add the "octave" elements
elements_settings = [q0, q1, q2, q3, q4, q5, q6, q7, q8, rr0, rr1, rr2, rr3, rr4, rr5, rr6, rr7, rr8]

###################
# Octave settings #
###################
# Configure the Octave according to the elements settings and calibrate
machine = QuAM('quam_state.json', flat_data=False)
config = build_config(machine)
qmm = QuantumMachinesManager(host=machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config, log_level="ERROR")
octave_settings(
    qmm=qmm,
    config=config,
    octaves=octaves,
    elements_settings=elements_settings,
    calibration=True,
)
qmm.close()
