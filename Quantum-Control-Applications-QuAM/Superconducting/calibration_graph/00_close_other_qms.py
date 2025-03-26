# %%
from typing import Optional, List

from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM


description = """
        CLOSE ALL OTHER QMs.
"""


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None


node = QualibrationNode(
    name="00_Close_other_QMs", description=description, parameters=Parameters()
)

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()
qmm.close_all_qms()
