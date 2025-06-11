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
    name="FluxTunableTransmon_Retuning",
    parameters=Parameters(),
    nodes={
        "IQ_blobs": library.nodes["07_iq_blobs"].copy(name="IQ_blobs"),
        "ramsey_vs_flux_calibration": library.nodes["09_ramsey_vs_flux_calibration"].copy(
            name="ramsey_vs_flux_calibration"
        ),
        "power_rabi_error_amplification_x180": library.nodes["04b_power_rabi"].copy(
            name="power_rabi_error_amplification_x180",
            max_number_pulses_per_sweep=200,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            use_state_discrimination=True,
        ),
        "power_rabi_error_amplification_x90": library.nodes["04b_power_rabi"].copy(
            name="power_rabi_error_amplification_x90",
            max_number_pulses_per_sweep=200,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            operation="x90",
            update_x90=False,
            use_state_discrimination=True,
        ),
        "Randomized_benchmarking": library.nodes["11a_single_qubit_randomized_benchmarking"].copy(
            name="Randomized_benchmarking",
            use_state_discrimination=True,
            delta_clifford=20,
            num_random_sequences=500,
        ),
    },
    connectivity=[
        ("IQ_blobs", "ramsey_vs_flux_calibration"),
        ("ramsey_vs_flux_calibration", "power_rabi_error_amplification_x180"),
        ("power_rabi_error_amplification_x180", "power_rabi_error_amplification_x90"),
        ("power_rabi_error_amplification_x90", "Randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
