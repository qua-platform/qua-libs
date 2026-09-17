# %%
from typing import List

from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.core.parameters import GraphParameters
from qualibrate import QualibrationGraph
from qualibrate import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name = "qubit_pairs"
    qubit_pairs: List[str] = ["D2-D4"]


g = QualibrationGraph(
    name="CZ_Calibration_Tunable_Couplers",
    parameters=Parameters(),
    nodes={
        "flux_bootstrap": library.nodes["30_cz_iswap_flux_bootstrap"].copy(name="flux_bootstrap"),
        "leakage": library.nodes["32a_cz_leakage_amplification"].copy(name="leakage"),
        "conditional_phase": library.nodes["33a_cz_conditional_phase"].copy(name="conditional_phase"),
        "conditional_phase_error_amp": library.nodes["33b_cz_conditional_phase_error_amp"].copy(
            name="conditional_phase_error_amp"
        ),
        "phase_compensation": library.nodes["34a_cz_phase_compensation"].copy(name="phase_compensation"),
    },
    connectivity=[
        ("flux_bootstrap", "leakage"),
        ("leakage", "conditional_phase"),
        ("conditional_phase", "conditional_phase_error_amp"),
        ("conditional_phase", "phase_compensation"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()

# %%
