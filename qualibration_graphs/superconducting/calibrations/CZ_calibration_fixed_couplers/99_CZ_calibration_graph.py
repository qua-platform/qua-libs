# %%
from typing import List

from calibration_utils.T1 import parameters
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name = "qubit_pairs"
    qubit_pairs: List[str] = ["D2-D4"]


g = QualibrationGraph(
    name="CZ_Calibration_Fixed_Couplers",
    parameters=Parameters(),
    nodes={
        "chevron": library.nodes["19_chevron_1102"].copy(name="chevron"),
        "conditional_phase": library.nodes["20_cz_conditional_phase"].copy(name="conditional_phase"),
        "phase_compensation": library.nodes["21_cz_phase_compensation"].copy(name="phase_compensation"),
    },
    connectivity=[
        ("chevron", "conditional_phase"),
        ("conditional_phase", "phase_compensation"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()

# %%
