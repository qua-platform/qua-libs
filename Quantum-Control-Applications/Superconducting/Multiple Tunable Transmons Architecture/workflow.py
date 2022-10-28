import entropynodes.library as nodes  # noqa: F401
from flame.workflow import Workflow

wf = Workflow("ResSpecDemo", description="res spec -> analysis")

# now start workflow description

res_spec = nodes.ResSpecNode("node_res_spec")
res_spec_analysis = nodes.ResSpecAnalysisNode("node_res_spec_analysis", IQ=res_spec.o.IQ)

# end of workflow description

wf.register()
