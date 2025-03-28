# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM


description = """
        CLOSE ALL OTHER QMs.
"""


node = QualibrationNode[NodeParameters, QuAM](
    name="00_close_other_qms", description=description, parameters=NodeParameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# Generate the OPX and Octave configurations
config = node.machine.generate_config()
# Open Communication with the QOP
qmm = node.machine.connect()
qmm.close_all_qms()
