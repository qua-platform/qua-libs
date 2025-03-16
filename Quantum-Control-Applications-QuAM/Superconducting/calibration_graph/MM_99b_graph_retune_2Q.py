# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from typing import ClassVar, Optional

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["qubitC4", "qubitC5"]


multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "active"


g = QualibrationGraph(
    name="retune_2Q_graph",
    parameters=Parameters(),
    nodes={
        "Cz_phase_calibration_frame": library.nodes["MM_32a_Cz_phase_calibration_frame"].copy(
            flux_point_joint_or_independent=flux_point,
            name="Cz_phase_calibration_frame",
            reset_type=reset_type_thermal_or_active,
        ),
        "Cz_1Q_phase_calibration_frame": library.nodes["MM_33a_Cz_1Qphase_calibration_frame"].copy(
            flux_point_joint_or_independent=flux_point,
            name="Cz_1Q_phase_calibration_frame",
            reset_type=reset_type_thermal_or_active,
        ),
        "2Q_confusion_matrix": library.nodes["MM_34_2Q_confusion_matrix"].copy(
            flux_point_joint_or_independent=flux_point,
            name="2Q_confusion_matrix",
            reset_type=reset_type_thermal_or_active,
        ),
        "Bell_state_tomography": library.nodes["MM_40b_Bell_state_tomography"].copy(
            flux_point_joint_or_independent=flux_point,
            name="Bell_state_tomography",
            reset_type=reset_type_thermal_or_active,
        ),
    },
    connectivity=[
        ("Cz_phase_calibration_frame", "Cz_1Q_phase_calibration_frame"),
        ("Cz_1Q_phase_calibration_frame", "2Q_confusion_matrix"),
        ("2Q_confusion_matrix", "Bell_state_tomography"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%
