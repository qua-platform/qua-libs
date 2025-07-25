# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_config import Quam


description = """
        CLOSE ALL OTHER QMs.
"""

node = QualibrationNode[NodeParameters, Quam](
    name="00_close_other_qms", description=description, parameters=NodeParameters()
)


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Close_all_quantum_machines}
@node.run_action()
def close_all_quantum_machines(node: QualibrationNode[NodeParameters, Quam]):
    """Closes all the opened quantum machines."""
    qmm = node.machine.connect()
    qmm.close_all_qms()
    qmm.clear_all_job_results()
