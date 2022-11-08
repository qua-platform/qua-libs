import entropynodes.library as Nodes  # noqa: F401
from flame.workflow import Workflow

wf = Workflow("Qubit calibration", description="graph for qubit calibration")

# now start workflow description

res_spec = Nodes.res_spec_node('res_spec_node')

# end of workflow description

wf.register()
