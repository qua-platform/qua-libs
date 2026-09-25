"""Demonstrate adaptive spectroscopy followed by power Rabi.

Adjust resonator spectroscopy parameters between iterations. Validate
qubit spectroscopy and refine failed results before power Rabi.
"""

# %% Imports

from typing import List, Optional
from qualibrate import QualibrationLibrary, QualibrationGraph, GraphParameters, QualibrationNode
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from calibration_utils.optimisers import (
    resolve_resspec_params,
    resolve_qspec_params,
    validate_qspec,
    validate_qspec_fwhm,
    resolve_qspec_params_advanced,
)

# To use a specific library path instead of the active library:
# library_path= "C:/Data/Code Development/Qualibration_IQCC/qualibrate_demo/calibrations/Adaptive_calibrations"
# library = QualibrationLibrary.get_active_library(library_path)


library = QualibrationLibrary.get_active_library()

# %% Graph parameters and file names

max_loops = 3  # Maximum spectroscopy refinement iterations
qubit_names = ["qA2"]  # default qubits
pair_names = ["qA2-qA1"]  # default qubit pairs


class CalibrationParameters(GraphParameters):
    """Define target qubits, qubit pairs, and the spectroscopy refinement limit."""

    qubits: List[str] = qubit_names
    qubit_pairs: List[str] = pair_names
    max_iterations: int = max_loops  # Limit for the refinement loop


graph_name = "Adaptive_Spectroscopy_Graph"

# Library node identifiers used to create graph-specific copies
res_spec_file = "02d_resonator_spectroscopy_adaptive"
res_spec_flux_file = "02c_resonator_spectroscopy_vs_flux"
qubit_spec_file = "03c_qubit_spectroscopy_adaptive"
qubit_spec_flux_file = "03b_qubit_spectroscopy_vs_flux"
rabi_chevron_file = "04a_rabi_chevron"
rabi_power_file = "04b_power_rabi"  # also used for x180, x90 tuning
readout_power_file = "08b_readout_power_optimization"
readout_freq_file = "08a_readout_frequency_optimization"
iq_blobs_file = "07_iq_blobs"
ramsey_flux_file = "09a_ramsey_vs_flux_calibration"

t1_file = "05_T1"
ramsey_file = "06a_ramsey"
t2echo_file = "06b_echo"
drag_calib_file = "10b_drag_calibration_180_minus_180"
rb_file = "11a_single_qubit_randomized_benchmarking"

# %% Graph structure

with QualibrationGraph.build(
    name=graph_name, parameters=CalibrationParameters(), orchestrator=BasicOrchestrator(skip_failed=False)
) as graph:

    # Start resonator spectroscopy with a 30 MHz frequency span.
    res_spec_node = library.nodes[res_spec_file].copy(
        name="resonator_spectroscopy", qubits=graph.parameters.qubits, frequency_span_in_mhz=30
    )

    # Use a high drive amplitude to demonstrate the failure-to-refinement branch.
    q_spec_node = library.nodes[qubit_spec_file].copy(
        name="qubit_spectroscopy", qubits=graph.parameters.qubits, operation_amplitude_factor=1.999
    )

    power_rabi_node = library.nodes[rabi_power_file].copy(
        name="power_rabi",
        qubits=graph.parameters.qubits,
    )

    # Use a separate spectroscopy node for retries with adjusted parameters.
    refine_qubit_spec_node = library.nodes[qubit_spec_file].copy(
        name="refine_qubit_spectroscopy",
        qubits=graph.parameters.qubits,
    )
    graph.add_node(res_spec_node)
    graph.add_node(q_spec_node)

    graph.add_node(refine_qubit_spec_node)
    graph.add_node(power_rabi_node)

    # Connect calibration stages and define adaptive retries.

    # Allow two resonator iterations, adapting the sweep with resolve_resspec_params.
    graph.loop(res_spec_node, max_iterations=2, resolve_params=resolve_resspec_params)

    graph.connect(res_spec_node, q_spec_node)

    # Validate the initial qubit spectrum to select the success or failure branch.
    graph.loop(q_spec_node, max_iterations=1, on=validate_qspec)

    # A valid initial spectrum leads directly to power Rabi.
    graph.connect(q_spec_node, power_rabi_node)

    # On failure, compute retry parameters and enter spectroscopy refinement.
    graph.connect_on_failure(q_spec_node, refine_qubit_spec_node, resolve_params=resolve_qspec_params)

    # Retry with adjusted parameters until the linewidth criterion or iteration limit is reached.
    graph.loop(
        refine_qubit_spec_node,
        max_iterations=graph.parameters.max_iterations,
        on=validate_qspec_fwhm,
        resolve_params=resolve_qspec_params,
    )
    # Rejoin the power Rabi stage after refinement.
    graph.connect(refine_qubit_spec_node, power_rabi_node)
