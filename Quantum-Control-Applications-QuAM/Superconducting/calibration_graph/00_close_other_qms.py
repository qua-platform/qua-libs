# %%
from typing import Optional, List
from qualibrate import QualibrationNode, NodeParameters

from quam_config import QuAM

description = """
    Close all other open QMs.
"""


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None


node = QualibrationNode(
    name="00_Close_other_QMs", parameters=Parameters(), description=description
)

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()
qmm.close_all_qms()
