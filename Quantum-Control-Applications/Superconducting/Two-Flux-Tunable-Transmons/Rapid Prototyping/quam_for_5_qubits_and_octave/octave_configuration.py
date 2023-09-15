"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
You need to run this file in order to update the Octaves with the new parameters.
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from quam import QuAM


config = build_config(machine)
# Configure the Octave parameters for each element
resonator = ElementsSettings(
    machine.resonators[active_qubits[0]].name, gain=-15, rf_in_port=["octave1", 1], down_convert_LO_source="Internal"
)
qubit_1 = ElementsSettings(machine.qubits[active_qubits[0]].name + "_xy", gain=2)
qubit_2 = ElementsSettings(machine.qubits[active_qubits[1]].name + "_xy", gain=2)
# Regroup all the elements
elements_settings = [resonator, qubit_1, qubit_2]

###################
# Octave settings #
###################
# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(
    host=machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config, log_level="ERROR"
)
octave_settings(
    qmm=qmm,
    config=config,
    octaves=[octave_1],
    elements_settings=elements_settings,
    calibration=False,
)
qmm.close_all_quantum_machines()
qmm.close()
