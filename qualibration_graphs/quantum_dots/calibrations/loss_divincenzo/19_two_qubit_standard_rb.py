"""Two-qubit standard randomized benchmarking calibration node.

This module implements a calibration node for performing standard randomized
benchmarking on two-qubit systems to measure average gate fidelity.
"""

# pylint: disable=duplicate-code

# %% {Imports}
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from more_itertools import flatten
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualibrate.core import QualibrationNode
from qualibrate.core.parameters import NodeParameters
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from quam_config import Quam

from calibration_utils.two_qubit_rb.parameters import Parameters
from calibration_utils.two_qubit_rb.rb_utils import StandardRB
from calibration_utils.two_qubit_rb.circuit_utils import (
    layerize_quantum_circuit,
    process_circuit_to_integers,
)
from calibration_utils.two_qubit_rb.qua_utils import QuaProgramHandler
from calibration_utils.two_qubit_rb.analysis import (
    log_fitted_results,
    process_raw_dataset,
)
from calibration_utils.two_qubit_rb.data_utils import RBResult
from calibration_utils.two_qubit_rb.plot_utils import gate_mapping
from calibration_utils.two_qubit_rb.rb_cache import cache_key, try_load, save
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot

# %% {Initialisation}
description = """
TWO-QUBIT STANDARD RANDOMIZED BENCHMARKING

The program consists in playing random sequences of Clifford gates and measuring the state of the resonators afterward.
Each random sequence is generated for the maximum depth (specified as an input) and played for each depth asked by the
user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate that will
bring the qubits back to their ground state.

The random circuits are generated offline as a sequence of Clifford gates and then transpiled to a basis gate set
(default is ['rz', 'sx', 'x', 'cz']).
The circuits are executed per two-qubit layer using a switch_case block structure, allowing for efficient execution
of the quantum circuits.

Standard randomized benchmarking provides a measure of the average gate fidelity by fitting the survival probability
to an exponential decay as a function of circuit depth. This gives an estimate of the overall gate error rate for
the two-qubit system.

Key Features:
    - use_input_stream: When enabled, the circuit sequences are streamed to the OPX by using the
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

Each sequence is played multiple times for averaging, and multiple random sequences are generated for each depth to
improve statistical significance. The data is then post-processed to extract the two-qubit Clifford fidelity.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates.
    - Having calibrated the two-qubit gate (cz) that will be used in the Clifford sequences.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization, IQ_blobs).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="19_two_qubit_standard_rb",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/two_qubit_rb/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.use_simulated_data = False

if node.parameters.use_input_stream:
    raise NotImplementedError("Input streams is not supported yet.")

# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "shots": xr.DataArray(np.arange(node.parameters.num_shots)),
        "depths": xr.DataArray(np.array(node.parameters.circuit_lengths)),
        "sequence": xr.DataArray(np.arange(node.parameters.num_circuits_per_length)),
    }

    ## try to load cached circuits
    key = cache_key(node.parameters.seed, node.parameters.circuit_lengths, node.parameters.num_circuits_per_length)
    cache_dir = Path(__file__).resolve().parents[2] / ".rb_cache"
    cached = try_load(cache_dir, key)
    
    if cached is not None:
        circuits_as_ints = cached["circuits_as_ints"]
        node.namespace["average_layers_per_clifford"] = cached["average_layers_per_clifford"]
        node.log(f"Loaded {len(circuits_as_ints)} cached RB circuits (key {key[:12]})")
    else:
        standard_RB = StandardRB(
            circuit_lengths=node.parameters.circuit_lengths,
            num_circuits_per_length=node.parameters.num_circuits_per_length,
            num_qubits=2,
            seed=node.parameters.seed
        )
        ## transpile cliffords to native gates
        transpiled_circuits = standard_RB.transpiled_circuits
        transpiled_circuits_as_ints = {}
        for l, circuits in transpiled_circuits.items():
            transpiled_circuits_as_ints[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]
        
        # to calculate the average number of 2q layers per Clifford
        node.namespace["average_layers_per_clifford"] = np.mean(
            [
                np.mean([len(circ) for circ in circuits]) / np.array(length + 1)
                for length, circuits in transpiled_circuits_as_ints.items()
                if length > 0
            ]
        )
        
        circuits_as_ints = []
        for circuits_per_len in transpiled_circuits_as_ints.values():
            for circuit in circuits_per_len:
                circuit_with_measurement = circuit + [66]  # readout
                # circuit_with_measurement = circuit + [38]  # readout
                circuits_as_ints.append(circuit_with_measurement)

        save(
            cache_dir,
            key,
            {
                "circuits_as_ints": circuits_as_ints,
                "average_layers_per_clifford": float(node.namespace["average_layers_per_clifford"]),
            },
        )
        node.log(f"Computed and cached {len(circuits_as_ints)} RB circuits (key {key[:12]})")

    num_pairs = len(qubit_pairs)
    
    qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)
    
    node.namespace["qua_program"] = qua_program_handler.get_qua_program()


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset
    called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset

# %%
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }

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
    """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    qubit_pairs = node.namespace["qubit_pairs"]
    node.results["fit_results"] = {}
    for qp in qubit_pairs:
    
        rb_result = RBResult(
            circuit_depths=list(node.parameters.circuit_lengths),
            num_repeats=node.parameters.num_circuits_per_length,
            num_averages=node.parameters.num_shots,
            state=node.results["ds_raw"].state.sel(qubit_pair=qp.name).data,
            average_layers_per_clifford=node.namespace["average_layers_per_clifford"],
        )
    
        fig = rb_result.plot_with_fidelity()
        fig.suptitle(f"2Q Randomized Benchmarking - {qp.name}")
        fig.show()
    
        node.results[f"fig_{qp.name}"] = fig
        node.results["fit_results"][qp.name] = {
            "success": rb_result.fit_success,
            "alpha": rb_result.alpha,
            "fidelity": rb_result.fidelity,
        }
    
        log_fitted_results(node.results["fit_results"], log_callable=node.log)
    
        node.outcomes = {
            qp_name: ("successful" if fit_result.get("success") else "failed")
            for qp_name, fit_result in node.results["fit_results"].items()
        }

    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant state entries when RB fitting succeeds."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            # Store RB metrics in the pair's gate_fidelity dict (Dict[str, Any]).
            # Note: the macro's own `fidelity` attribute is typed Optional[float],
            # so it must not be used to hold a dict of metrics.
            qubit_pair = node.machine.qubit_pairs[qp.name]
            operation = node.parameters.operation
            qubit_pair.gate_fidelity[operation] = {
                "StandardRB": node.results["fit_results"][qp.name]["fidelity"],
                "StandardRB_alpha": node.results["fit_results"][qp.name]["alpha"],
                "StandardRB_load_id": node.snapshot_idx,
            }

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state snapshot."""
    node.save()


# %%
