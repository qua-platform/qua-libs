"""
A simple program to calibrate Octave mixers for all qubits and resonators
"""

from pathlib import Path
from quam_libs.components import QuAM

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Instantiate the QuAM class from the state file

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
qm = qmm.open_qm(config)

for qubit in machine.active_qubits:
    qubit.calibrate_octave(qm)
