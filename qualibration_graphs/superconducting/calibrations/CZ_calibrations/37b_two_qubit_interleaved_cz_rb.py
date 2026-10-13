# pylint: disable=duplicate-code

# %% {Imports}
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from calibration_utils.two_qubit_rb import (
    InterleavedRB,
    Parameters,
    QuaProgramHandler,
    build_sweep_axes,
    cache_key,
    circuit_to_layer_ints,
    fit_raw_data,
    log_irb_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    rb_progress_total,
    RBMode,
    save,
    stamp_execution_format,
    log_depth_summary,
    summarize_transpiled_depth,
    try_load,
)
from calibration_utils.two_qubit_rb.rb_cache import DEFAULT_RB_BASIS_GATES

# %% {Initialisation}
description = """
        TWO-QUBIT INTERLEAVED CZ RANDOMIZED BENCHMARKING
Interleaved randomised benchmarking isolates the error of a specific target gate (here the CZ) by
running two experiments: a reference RB sequence of random Cliffords, and an interleaved sequence in
which the target gate is inserted between every random Clifford. Fitting both survival-probability
decays as a function of circuit depth and taking the ratio of the two decay rates yields the average
error of the target gate, independent of the surrounding Clifford errors. Each random sequence is
generated offline as a Clifford sequence, transpiled to the analog-XY basis (default ['cz', 'sx',
'x', 'ry', 'y']), and executed per two-qubit layer using a gate-only switch_case (opcodes 0–37) with
analog X/Y plays. Readout, state saving, reset, and frame cleanup run once per circuit outside that
switch. Sequences are truncated to each requested depth and end with a recovery gate that returns
the qubits to the ground state. Each sequence is played multiple times for averaging, and multiple
random sequences are generated per depth for statistical significance. The data is post-processed to
extract the CZ gate error and fidelity. IRB vs a pre-XY (ZX-basis / virtual-Z) 37a overlay is not
valid; rerun 37a on analog XY first. EPG is not comparable across those bases; EPC remains the RB
observable.

Key Features:
    - use_input_stream: When enabled, the circuit sequences are streamed to the OPX by using the
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates (including y90 / y180 / -y90).
    - Having calibrated the two-qubit gate (cz) that will be used in the Clifford sequences.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.
    - Having a recent reference Standard RB run (37a_two_qubit_standard_rb) for the same operation,
      acquired with analog-XY encoding v3 (basis ['cz', 'sx', 'x', 'ry', 'y']). Rerun 37a before this
      node if the saved StandardRB overlay is ZX-basis (virtual Z) or predates the unsafe-switch
      timing fix; do not mix a ZX 37a reference with an XY 37b acquisition.

State update:
    - qp.macros[operation].fidelity["InterleavedRB"] (fitted CZ-gate fidelity).
    - qp.macros[operation].fidelity["InterleavedRB_alpha"] (fitted RB decay parameter).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="37b_two_qubit_interleaved_cz_rb",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
    machine=Quam.load(),  # Instantiate the QUAM class from the state file
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    if node.parameters.interval_spacing == "linear":
        circuit_depths = np.unique(
            np.linspace(0, node.parameters.max_circuit_depth, node.parameters.num_intervals, dtype=int)
        ).tolist()
    else:
        circuit_depths = np.unique(
            np.geomspace(1, node.parameters.max_circuit_depth + 1, node.parameters.num_intervals, dtype=int) - 1
        ).tolist()
    node.namespace["circuit_depths"] = circuit_depths
    node.log(f"Circuit depths: {circuit_depths}")

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    node.namespace["sweep_axes"] = build_sweep_axes(
        qubit_pairs,
        node.parameters.num_shots,
        circuit_depths,
        node.parameters.num_circuits_per_depth,
        use_input_stream=node.parameters.use_input_stream,
    )

    key = cache_key(
        node.parameters.seed,
        circuit_depths,
        node.parameters.num_circuits_per_depth,
        target_gate="cz",
        basis_gates=DEFAULT_RB_BASIS_GATES,
    )
    cache_dir = Path(__file__).resolve().parents[2] / ".rb_cache"
    cached = try_load(cache_dir, key)

    if cached is not None:
        circuits_as_ints = cached["circuits_as_ints"]
        node.log(f"Loaded {len(circuits_as_ints)} cached interleaved RB circuits")
        if "depth_summaries" in cached:
            for summary in cached["depth_summaries"]:
                log_depth_summary(summary, log_callable=node.log)
    else:
        interleaved_RB = InterleavedRB(
            target_gate="cz",
            amplification_lengths=circuit_depths,
            num_circuits_per_length=node.parameters.num_circuits_per_depth,
            basis_gates=list(DEFAULT_RB_BASIS_GATES),
            num_qubits=2,
            seed=node.parameters.seed,
        )

        transpiled_circuits = interleaved_RB.transpiled_circuits
        transpiled_circuits_as_ints = {}
        depth_summaries = []
        total_circuits = sum(len(c) for c in transpiled_circuits.values())
        with tqdm(total=total_circuits, desc="Encoding RB circuits to ints", unit="circ") as pbar:
            for l, circuits in transpiled_circuits.items():
                encoded = []
                for qc in circuits:
                    encoded.append(circuit_to_layer_ints(qc))
                    pbar.update(1)
                transpiled_circuits_as_ints[l] = encoded
                depth_summaries.append(summarize_transpiled_depth(l, circuits, log_callable=node.log))

        circuits_as_ints = []
        for circuits_per_len in transpiled_circuits_as_ints.values():
            for circuit in circuits_per_len:
                circuits_as_ints.append(circuit)

        save(cache_dir, key, {"circuits_as_ints": circuits_as_ints, "depth_summaries": depth_summaries})
        node.log(f"Computed and cached {len(circuits_as_ints)} interleaved RB circuits")

    num_pairs = len(qubit_pairs)

    qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)

    node.namespace["qua_program_handler"] = qua_program_handler
    node.namespace["qua_program"] = qua_program_handler.get_qua_program()


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data
    and store it in a xarray dataset called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Feed the input-stream queue upfront: pair-major, then sub-chunk
        # (one host push per advance; shots replay each chunk on the OPX).
        if node.parameters.use_input_stream:
            node.namespace["qua_program_handler"].push_all_chunks(job)
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        progress_total = rb_progress_total(
            node.parameters.num_shots,
            node.parameters.num_circuits_per_depth,
            len(node.namespace["circuit_depths"]),
            use_input_stream=node.parameters.use_input_stream,
        )
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                progress_total,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset and the sweep/role data needed for reproducible re-analysis
    node.results["ds_raw"] = stamp_execution_format(dataset, node)


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["circuit_depths"] = node.results["ds_raw"].circuit_depth.values.tolist()


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse raw data, fit, log results, set outcomes and store structured fit results."""
    node.results["ds_proc"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_proc"], node, mode=RBMode.INTERLEAVED)
    if "rb_execution_format_version" in node.results["ds_raw"].attrs:
        stamp_execution_format(node.results["ds_proc"], node)
        stamp_execution_format(node.results["ds_fit"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_irb_results(fit_results, log_callable=node.log)

    threshold = node.parameters.fidelity_threshold
    outcomes: dict[str, str] = {}
    for qp_name, fit_result in fit_results.items():
        if not fit_result.success:
            outcomes[qp_name] = "failed"
            continue
        if threshold is not None and fit_result.fidelity < threshold:
            node.log(
                f"Qubit pair {qp_name}: CZ fidelity {fit_result.fidelity * 100:.3f}% "
                f"below threshold {threshold * 100:.3f}% -> marking as failed."
            )
            outcomes[qp_name] = "failed"
            continue
        outcomes[qp_name] = "successful"
    node.outcomes = outcomes


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    figures = plot_raw_data_with_fit(node, title_prefix="2Q Interleaved CZ Randomized Benchmarking")
    for fig in figures.values():
        plt.show()
    node.results["figures"] = figures


# %% {Update_state}
with node.record_state_updates():
    for qp in node.namespace["qubit_pairs"]:
        if node.outcomes[qp.name] == "failed":
            continue
        node.machine.qubit_pairs[qp.name].macros[node.parameters.operation].fidelity["InterleavedRB"] = node.results[
            "fit_results"
        ][qp.name]["fidelity"]
        node.machine.qubit_pairs[qp.name].macros[node.parameters.operation].fidelity["InterleavedRB_alpha"] = (
            node.results["fit_results"][qp.name]["alpha"]
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results to the node."""
    node.save()


# %%
