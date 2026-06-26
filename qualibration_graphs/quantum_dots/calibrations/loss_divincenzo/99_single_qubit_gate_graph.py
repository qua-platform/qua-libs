# %%
from typing import ClassVar, List, Optional

from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.core.parameters import GraphParameters
from qualibrate.core.qualibration_graph import QualibrationGraph
from qualibrate.core.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name: ClassVar[str | None] = "qubit_names"
    qubit_names: Optional[List[str]] = ['q1', 'q2']

g = QualibrationGraph(
    name="SingleQubitGateCalibration",
    parameters=Parameters(),
    nodes={
        # "psb_search_opx_fixed_detuning": library.nodes["06d_PSB_search_opx_fixed_detuning"].copy(
        #     name="psb_search_opx_fixed_detuning"
        # ),
        "qubit_spectroscopy": library.nodes["08b_qubit_spectroscopy"].copy(
            name="qubit_spectroscopy"
        ),
        "power_rabi": library.nodes["09a_power_rabi"].copy(
            name="power_rabi"
        ),
        "ramsey": library.nodes["11a_ramsey"].copy(
            name="ramsey"
        ),
        "power_rabi_error_amplification": library.nodes[
            "09b_power_rabi_error_amplification"
        ].copy(name="power_rabi_error_amplification"),
    },
    connectivity=[
        # ("psb_search_opx_fixed_detuning", "qubit_spectroscopy"),
        ("qubit_spectroscopy", "power_rabi"),
        ("power_rabi", "ramsey"),
        ("ramsey", "power_rabi_error_amplification"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
