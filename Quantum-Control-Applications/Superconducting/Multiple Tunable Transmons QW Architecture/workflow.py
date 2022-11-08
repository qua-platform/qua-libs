import entropynodes.library as Nodes  # noqa: F401
from flame.workflow import Workflow

wf = Workflow("Qubit calibration", description="graph for qubit calibration")

# now start workflow description

res_spec = Nodes.res_spec_node('res_spec_node')
res_spec_flux = Nodes.res_spec_flux('res_spec_flux_node', state=res_spec.o.state, debug=res_spec.o.debug, resources=res_spec.o.resources)

# end of workflow description

wf.register()
