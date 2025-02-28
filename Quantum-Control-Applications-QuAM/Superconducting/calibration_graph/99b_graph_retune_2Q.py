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
    qubit_pairs: List[str] = None
    targets_name: ClassVar[Optional[str]] = "qubit_pairs"

multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "active"


g = QualibrationGraph(
    name="retune_2Q_graph",
    parameters=Parameters(),
    nodes={
        "Cz_phase_calibration_frame": library.nodes["32a_Cz_phase_calibration_frame"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="Cz_phase_calibration_frame",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),
        "Cz_1Q_phase_calibration_frame": library.nodes["33a_Cz_1Qphase_calibration_frame"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="Cz_1Q_phase_calibration_frame",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),
        "2Q_confusion_matrix": library.nodes["34_2Q_confusion_matrix"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="2Q_confusion_matrix",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),   
        "Bell_state_tomography": library.nodes["40b_Bell_state_tomography"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="Bell_state_tomography",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
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

g.run(qubit_pairs=["qC4-qC5"])
# %%
