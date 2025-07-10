# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["qubitC1", "qubitC2", "qubitC3"]



multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "active"



g = QualibrationGraph(
    name="gate_cal_graph",
    parameters=Parameters(),
    nodes={
        "power_rabi": library.nodes["04_Power_Rabi"].copy(flux_point_joint_or_independent=flux_point, 
                                                          multiplexed=multiplexed, 
                                                          name="power_rabi"),
        "ramsey": library.nodes["06_Ramsey"].copy(flux_point_joint_or_independent=flux_point, 
                                                  multiplexed=multiplexed, 
                                                  name="ramsey"),
        "drag_calibration": library.nodes["09b_DRAG_Calibration_180_minus_180"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                     multiplexed=multiplexed, 
                                                                                     name="drag_calibration"),
        "stark_detuning_calibration": library.nodes["09a_Stark_Detuning"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                         multiplexed=multiplexed, 
                                                                                         name="stark_detuning_calibration"),
        "power_rabi_error_amplification_x180": library.nodes["04_Power_Rabi"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                   multiplexed=multiplexed, operation_x180_or_any_90 = "x180", 
                                                                                   min_amp_factor = 0.9, max_amp_factor = 1.1, amp_factor_step = 0.005,
                                                                                   max_number_rabi_pulses_per_sweep = 200, reset_type_thermal_or_active = reset_type_thermal_or_active, 
                                                                                   state_discrimination = True, update_x90 = True,
                                                                                   name="power_rabi_error_amplification_x180"),
        "power_rabi_error_amplification_x90": library.nodes["04_Power_Rabi"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                  multiplexed=multiplexed, operation_x180_or_any_90 = "x90", 
                                                                                   min_amp_factor = 0.95, max_amp_factor = 1.05, amp_factor_step = 0.005,
                                                                                   max_number_rabi_pulses_per_sweep = 200, reset_type_thermal_or_active = reset_type_thermal_or_active, 
                                                                                   state_discrimination = True, update_x90 = True,
                                                                                   name="power_rabi_error_amplification_x90"),
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                                               multiplexed=multiplexed, 
                                                                                                               name="single_qubit_randomized_benchmarking"),
    },
    connectivity=[
                  ("power_rabi", "ramsey"),
                  ("ramsey", "drag_calibration"),
                  ("drag_calibration", "stark_detuning_calibration"),
                  ("stark_detuning_calibration", "power_rabi_error_amplification_x180"),
                  ("power_rabi_error_amplification_x180", "power_rabi_error_amplification_x90"),
                  ("power_rabi_error_amplification_x90", "single_qubit_randomized_benchmarking"),
                  ],

    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%
