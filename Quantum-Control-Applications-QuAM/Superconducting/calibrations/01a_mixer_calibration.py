"""
A simple program to calibrate Octave mixers for all qubits and resonators
"""
from pathlib import Path
from quam_components import QuAM

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Instantiate the QuAM class from the state file

# Define a path relative to this script, i.e., ../configuration/quam_state
config_path = Path(__file__).parent.parent / "configuration" / "quam_state"
# Load the machine state
machine = QuAM.load(config_path)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
qm = qmm.open_qm(config)

for qubit in machine.active_qubits:
    qubit.calibrate_octave(qm)
