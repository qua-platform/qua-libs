import entropynodes.library as nodes  # noqa: F401
from flame.workflow import Workflow

wf = Workflow("ResSpecDemo", description="res spec -> res spec analysis -> qubit spec")

# now start workflow description

res_spec = nodes.ResSpecNode("node_res_spec")
res_spec_analysis = nodes.ResSpecAnalysisNode("node_res_spec_analysis", IQ=res_spec.o.IQ, state=res_spec.o.state)
qb_spec = nodes.QbSpecNode("node_qubit_spec", state=res_spec_analysis.o.state)

# end of workflow description

wf.register()
