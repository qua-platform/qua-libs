# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["qubitC4", "qubitC5"]


multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "active"


g = QualibrationGraph(
    name="retune_fine_1Q_2Q_graph",
    parameters=Parameters(),
    nodes={
        "IQ_blobs": library.nodes["MM_07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="IQ_blobs",
            reset_type_thermal_or_active="thermal",
        ),
        "ramsey_flux_calibration": library.nodes["MM_08_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent=flux_point, multiplexed=multiplexed, name="ramsey_flux_calibration"
        ),
        "power_rabi_x180": library.nodes["MM_04_Power_Rabi"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=200,
            update_x90=False,
            state_discrimination=True,
        ),
        "power_rabi_x90": library.nodes["MM_04_Power_Rabi"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            operation_x180_or_any_90="x90",
            name="power_rabi_x90",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=200,
            state_discrimination=True,
        ),
        "single_qubit_randomized_benchmarking": library.nodes["MM_10a_Single_Qubit_Randomized_Benchmarking"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=False,
            delta_clifford=25,
            max_circuit_depth=400,
            num_random_sequences=500,
            name="single_qubit_randomized_benchmarking",
        ),
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
        ("IQ_blobs", "ramsey_flux_calibration"),
        ("ramsey_flux_calibration", "power_rabi_x180"),
        ("power_rabi_x180", "power_rabi_x90"),
        ("power_rabi_x90", "single_qubit_randomized_benchmarking"),
        ("single_qubit_randomized_benchmarking", "Cz_phase_calibration_frame"),
        ("Cz_phase_calibration_frame", "Cz_1Q_phase_calibration_frame"),
        ("Cz_1Q_phase_calibration_frame", "2Q_confusion_matrix"),
        ("2Q_confusion_matrix", "Bell_state_tomography"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%
