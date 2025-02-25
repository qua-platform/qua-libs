# %%

"""
A simple program to close all other open QMs.
"""

from typing import Optional, List
from qualibrate import QualibrationNode, NodeParameters

from quam_config import QuAM


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None


node = QualibrationNode(name="00_Close_other_QMs", parameters=Parameters())

# Instantiate the QuAM class from the state file
machine = QuAM.load("C:\git\qua-libs\Quantum-Control-Applications-QuAM\Superconducting\configuration\quam_state")

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()
qmm.close_all_qms()
