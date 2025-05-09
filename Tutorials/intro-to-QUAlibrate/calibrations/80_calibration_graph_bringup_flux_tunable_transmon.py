# %%
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["q1"]


g = QualibrationGraph(
    name="CalibrationGraphExample",
    parameters=Parameters(),
    nodes={
        "node_1": library.nodes["node_1"].copy(name="node_1"),
        "node_2": library.nodes["node_2"].copy(name="node_2"),
    },
    connectivity=[
        ("node_1", "node_2"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
