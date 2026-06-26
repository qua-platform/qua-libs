# %%
from typing import ClassVar, List, Optional

from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.core.parameters import GraphParameters
from qualibrate.core.qualibration_graph import QualibrationGraph
from qualibrate.core.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name: ClassVar[str | None] = "qubit_pairs"
    qubit_pairs: Optional[List[str]] = ["q1_q2"]


g = QualibrationGraph(
    name="InitGateOptimization",
    parameters=Parameters(),
    nodes={
        "optimize_initialization": library.nodes["02_optimize_initialization"].copy(
            name="optimize_initialization"
        ),
        "cmaes_gate_optimization": library.nodes["03a_cmaes_gate_optimization"].copy(
            name="cmaes_gate_optimization"
        ),
    },
    connectivity=[
        ("optimize_initialization", "cmaes_gate_optimization"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
