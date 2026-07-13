# pylint: disable=duplicate-code

# %% {Imports}
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
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
    Parameters,
    QuaProgramHandler,
    StandardRB,
    build_sweep_axes,
    READOUT_OPCODE,
    cache_key,
    circuit_to_layer_ints,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    save,
    log_depth_summary,
    summarize_transpiled_depth,
    try_load,
)

# %% {Initialisation}
description = """
        TWO-QUBIT STANDARD RANDOMIZED BENCHMARKING
The program plays random sequences of Clifford gates on the qubit pair and measures the resulting
state of the resonators. Each random sequence is generated for the maximum depth (specified as an
input) and played for every depth requested by the user (the sequence is truncated to the desired
depth). Each truncated sequence ends with the recovery gate that brings the qubits back to their
ground state. The random circuits are generated offline as Clifford sequences and then transpiled to
a basis gate set (default ['rz', 'sx', 'x', 'cz']); they are executed per two-qubit layer using a
switch_case block for efficient execution. Each sequence is played multiple times for averaging, and
multiple random sequences are generated per depth for statistical significance. Standard RB measures
the average two-qubit Clifford fidelity by fitting the survival probability to an exponential decay
as a function of circuit depth.

Key Features:
    - use_input_stream: When enabled, the circuit sequences are streamed to the OPX by using the
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates.
    - Having calibrated the two-qubit gate (cz) that will be used in the Clifford sequences.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.

State update:
    - qp.macros[operation].fidelity["StandardRB"] (fitted average Clifford fidelity).
    - qp.macros[operation].fidelity["StandardRB_alpha"] (fitted RB decay parameter).
    - qp.macros[operation].fidelity["StandardRB_load_id"] (snapshot index of this run).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="37a_two_qubit_standard_rb",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
    machine=Quam.load(),  # Instantiate the QUAM class from the state file
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    node.namespace["sweep_axes"] = build_sweep_axes(
        qubit_pairs,
        node.parameters.num_shots,
        node.parameters.circuit_depths,
        node.parameters.num_circuits_per_depth,
        use_input_stream=node.parameters.use_input_stream,
    )

    key = cache_key(
        node.parameters.seed, node.parameters.circuit_depths, node.parameters.num_circuits_per_depth
    )  # key to cache the RB circuits
    cache_dir = Path(__file__).resolve().parents[2] / ".rb_cache"
    cached = try_load(cache_dir, key)  # try to load the cached RB circuits

    if cached is not None:  # if the cached RB circuits are found, load them
        circuits_as_ints = cached["circuits_as_ints"]
        node.namespace["average_gates_per_clifford"] = cached["average_gates_per_clifford"]
        node.namespace["avg_1q_per_clifford"] = cached["avg_1q_per_clifford"]
        node.namespace["avg_cz_per_clifford"] = cached["avg_cz_per_clifford"]
        node.log(f"Loaded {len(circuits_as_ints)} cached RB circuits (key {key[:12]})")
        if "depth_summaries" in cached:
            for summary in cached["depth_summaries"]:
                log_depth_summary(summary, log_callable=node.log)
    else:
        standard_RB = StandardRB(  # if the cached RB circuits are not found, generate them
            amplification_lengths=node.parameters.circuit_depths,
            num_circuits_per_length=node.parameters.num_circuits_per_depth,
            num_qubits=2,
            seed=node.parameters.seed,
        )

        transpiled_circuits = standard_RB.transpiled_circuits  # transpile the circuits
        transpiled_circuits_as_ints = {}
        depth_summaries = []  # summarize the transpiled circuits and save them to the cached dictionary
        total_circuits = sum(len(c) for c in transpiled_circuits.values())  # count the total number of circuits
        with tqdm(total=total_circuits, desc="Encoding RB circuits to ints", unit="circ") as pbar:
            for l, circuits in transpiled_circuits.items():
                encoded = []  # encode the circuits to integers
                for qc in circuits:
                    encoded.append(circuit_to_layer_ints(qc))
                    pbar.update(1)
                transpiled_circuits_as_ints[l] = encoded  # save the encoded circuits to the dictionary
                depth_summaries.append(
                    summarize_transpiled_depth(l, circuits, log_callable=node.log)
                )  # save the depth summaries to the list

        total_cliffords = sum(s["total_cliffords"] for s in depth_summaries)
        node.namespace["average_gates_per_clifford"] = (
            sum(s["total_transpiled_gates"] for s in depth_summaries) / total_cliffords if total_cliffords else 0.0
        )
        node.namespace["avg_1q_per_clifford"] = (
            sum(s["total_1q_gates"] for s in depth_summaries) / total_cliffords if total_cliffords else 0.0
        )
        node.namespace["avg_cz_per_clifford"] = (
            sum(s["total_cz_gates"] for s in depth_summaries) / total_cliffords if total_cliffords else 0.0
        )

        circuits_as_ints = []  # encode the circuits to integers
        for circuits_per_len in transpiled_circuits_as_ints.values():
            for circuit in circuits_per_len:
                circuit_with_measurement = circuit + [READOUT_OPCODE]
                circuits_as_ints.append(circuit_with_measurement)

        save(
            cache_dir,  # save the cached dictionary to the cache directory
            key,  # save the key to the cache directory
            {
                "circuits_as_ints": circuits_as_ints,  # save the encoded circuits to the cached dictionary
                "average_gates_per_clifford": float(node.namespace["average_gates_per_clifford"]),
                "avg_1q_per_clifford": float(node.namespace["avg_1q_per_clifford"]),
                "avg_cz_per_clifford": float(node.namespace["avg_cz_per_clifford"]),
                "depth_summaries": depth_summaries,  # save the depth summaries to the cached dictionary
            },
        )  # save the cached dictionary to the cache directory
        node.log(
            f"Computed and cached {len(circuits_as_ints)} RB circuits (key {key[:12]})"
        )  # log the number of circuits and the key

    num_pairs = len(qubit_pairs)  # count the number of qubit pairs

    qua_program_handler = QuaProgramHandler(
        node, num_pairs, circuits_as_ints, node.machine, qubit_pairs
    )  # create the QUA program handler

    node.namespace["qua_program_handler"] = qua_program_handler  # save the QUA program handler to the node namespace
    node.namespace["qua_program"] = qua_program_handler.get_qua_program()  # save the QUA program to the node namespace


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

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])  # execute the QUA program
        # Feed the input-stream queue upfront: pair-major, then sub-chunk
        # (one host push per advance; shots replay each chunk on the OPX).
        if node.parameters.use_input_stream:
            node.namespace["qua_program_handler"].push_all_chunks(job)  # push the chunks to the input-stream queue
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])  # create the data fetcher
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse raw data, fit, log results, set outcomes and store structured fit results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fit_result.success else "failed") for qp_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    figures = plot_raw_data_with_fit(node, title_prefix="2Q Standard Randomized Benchmarking")
    for fig in figures.values():
        plt.show()
    node.results["figures"] = figures


# %% {Update_state}
with node.record_state_updates():
    for qp in node.namespace["qubit_pairs"]:
        if node.outcomes[qp.name] == "failed":
            continue
        node.machine.qubit_pairs[qp.name].macros[node.parameters.operation].fidelity["StandardRB"] = node.results[
            "fit_results"
        ][qp.name]["fidelity"]
        node.machine.qubit_pairs[qp.name].macros[node.parameters.operation].fidelity["StandardRB_alpha"] = node.results[
            "fit_results"
        ][qp.name]["alpha"]
        node.machine.qubit_pairs[qp.name].macros[node.parameters.operation].fidelity[
            "StandardRB_load_id"
        ] = node.snapshot_idx


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results to the node."""
    node.save()


# %%
