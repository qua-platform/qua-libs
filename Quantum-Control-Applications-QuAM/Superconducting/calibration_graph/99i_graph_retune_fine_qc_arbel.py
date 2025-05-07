# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
from quam_libs.components.quam_root import QuAM
from quam_libs.lib.iqcc_cloud_data_storage_utils.upload_state_and_wiring import save_quam_state_to_cloud

# %%
library = QualibrationLibrary.get_active_library()

# %%
class Parameters(GraphParameters):
    qubits: List[str] = None

parameters = Parameters()

# Get the relevant QuAM components
if parameters.qubits is None:
    machine = QuAM.load()
    parameters.qubits = [q.name for q in machine.active_qubits]


multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="graph_retune_fine_qc_arbel",
    parameters=parameters,
    nodes={
        "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent=flux_point,
            multiplexed=multiplexed,
            name="IQ_blobs",
            reset_type_thermal_or_active="thermal",
        ),
        "ramsey_flux_calibration": library.nodes["08_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent=flux_point, multiplexed=multiplexed, name="ramsey_flux_calibration",
            flux_span = 0.016,
            flux_step = 0.0008,
            frequency_detuning_in_mhz = 4
        ),
        "power_rabi_x180": library.nodes["04_Power_Rabi"].copy(
            flux_point_joint_or_independent=flux_point,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=50,
            update_x90=True,
            state_discrimination=False,
            multiplexed=multiplexed,
        ),
        # "readout_power_optimization": library.nodes["07c_Readout_Power_Optimization"].copy(name="readout_power_optimization",
        # start_amp = 0.7,
        # end_amp = 1.3,
        # num_amps = 10
        # ),
        # "power_rabi_x90": library.nodes["04_Power_Rabi"].copy(
        #     flux_point_joint_or_independent=flux_point,
        #     multiplexed=multiplexed,
        #     operation_x180_or_any_90="x90",
        #     name="power_rabi_x90",
        #     reset_type_thermal_or_active=reset_type_thermal_or_active,
        #     min_amp_factor=0.8,
        #     max_amp_factor=1.2,
        #     amp_factor_step=0.002,
        #     max_number_rabi_pulses_per_sweep=50,
        #     state_discrimination=False,
        # ),
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            flux_point_joint_or_independent=flux_point, 
            multiplexed=True, 
            delta_clifford=100,
            num_random_sequences=1000,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            name="single_qubit_randomized_benchmarking"
        ),
    },
    connectivity=[
        ("IQ_blobs", "ramsey_flux_calibration"),
        ("ramsey_flux_calibration", "power_rabi_x180"),
        ("power_rabi_x180", "single_qubit_randomized_benchmarking"),
        # ("power_rabi_x180", "power_rabi_x90"),
        # ("power_rabi_x180", "readout_power_optimization"),
        # ("readout_power_optimization","single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%

save_quam_state_to_cloud(as_new_parent=False)
# %%
