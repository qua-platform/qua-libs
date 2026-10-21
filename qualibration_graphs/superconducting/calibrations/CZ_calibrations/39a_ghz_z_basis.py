"""GHZ Z-basis population measurement calibration node."""

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from calibration_utils.n_qubit_confusion_matrix import (
    get_qubit_groups,
    require_adjacent_cz_macros,
)
from calibration_utils.ghz_z_basis import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_ghz_z_basis,
)

# %% {Initialisation}
description = """
This experiment measures the Z-basis population distribution of N-qubit GHZ states (3 to 5 qubits).

Topology: GHZ preparation uses a linear CZ ladder (pair_01, pair_12, … in list order). This is not
a general graph builder — order qubit_groups so each consecutive pair exists on the machine (e.g.
star couplers qD2–qD1 and qD3–qD1 require ["qD2-qD1-qD3"], hub in the middle).

Parameters:
- qubit_groups: dash-separated qubit names per chain, e.g. ["qC4-qC3-qC2"] (order matters)

Process:
1. Prepare an N-qubit GHZ state using nearest-neighbor CZ gates
2. Perform simultaneous readout on all qubits
3. Apply readout error mitigation and compute the population distribution

Readout mitigation:
- Kron: tensor product of per-qubit resonator confusion matrices
- NQ: full N-qubit confusion matrix from node 38_n_qubit_confusion_matrix

Primary metric: Z-basis population fidelity = P(|0...0⟩) + P(|1...1⟩) after mitigation.

Prerequisites:
- Calibrated single-qubit gates and readout for all qubits in each chain
- Available nearest-neighbor CZ operations along the chain
- N-qubit confusion matrix in qubit pair extras for NQ mitigation (node 38)
"""

node = QualibrationNode[Parameters, Quam](
    name="39a_ghz_z_basis",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes only."""
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for GHZ Z-basis measurement."""
    node.namespace["qubit_groups"] = qubit_groups = get_qubit_groups(
        node, min_qubits=3, max_qubits=5, resolve_adjacent_pairs=True
    )
    operation = node.parameters.operation
    require_adjacent_cz_macros(qubit_groups, operation)

    num_qubit_groups = len(qubit_groups)
    num_qubits = qubit_groups[0].num_qubits
    n_shots = node.parameters.num_shots

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([qg.name for qg in qubit_groups]),
        "n": xr.DataArray(
            np.arange(n_shots),
            attrs={"long_name": "shot index"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        state_vars = [declare(int) for _ in range(num_qubits)]
        state_st_vars = [declare_output_stream() for _ in range(num_qubits)]
        state = [declare(int) for _ in range(num_qubit_groups)]
        state_st = [declare_output_stream() for _ in range(num_qubit_groups)]

        for group_idx, qg in enumerate(qubit_groups):
            for q in qg.qubits:
                node.machine.initialize_qpu(target=q)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)
                for q in qg.qubits:
                    q.reset(node.parameters.reset_type, node.parameters.simulate)
                align()

                qg.qubits[0].xy.play("-y90")
                qg.qubits[1].xy.play("y90")
                qg.qubit_pairs["pair_01"].macros[operation].apply()
                qg.qubits[0].xy.play("y90")
                for qubit_idx in range(2, num_qubits):
                    align()
                    qg.qubits[qubit_idx].xy.play("y90")
                    qg.qubit_pairs[f"pair_{qubit_idx - 1}{qubit_idx}"].macros[operation].apply()
                    qg.qubits[qubit_idx].xy.play("-y90")

                align()
                for idx, q in enumerate(qg.qubits):
                    q.readout_state(state_vars[idx])
                    save(state_vars[idx], state_st_vars[idx])

                state_expr = state_vars[0]
                for idx in range(1, num_qubits):
                    state_expr = state_expr * 2 + state_vars[idx]
                assign(state[group_idx], state_expr)
                save(state[group_idx], state_st[group_idx])
            align()

        with stream_processing():
            n_st.save("n")
            for group_idx in range(num_qubit_groups):
                state_st[group_idx].buffer(n_shots).save(f"state{group_idx + 1}")


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
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_groups"] = get_qubit_groups(node, min_qubits=3, max_qubits=5, resolve_adjacent_pairs=True)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Compute mitigated Z-basis populations and fidelities."""
    corrected_kron, corrected_nq, fit_results = fit_raw_data(node.results["ds_raw"], node)

    node.results["corrected_results"] = {k: v.tolist() for k, v in corrected_kron.items()}
    if corrected_nq:
        node.results["corrected_results_nq"] = {k: v.tolist() for k, v in corrected_nq.items()}
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    for qg in node.namespace["qubit_groups"]:
        fr = fit_results[qg.name]
        node.results["fidelities"] = node.results.get("fidelities", {})
        node.results["fidelities"][qg.name] = fr.fidelity_kron
        node.results[f"{qg.name}_fidelity_kron"] = fr.fidelity_kron
        if fr.fidelity_nq is not None:
            node.results["fidelities_nq"] = node.results.get("fidelities_nq", {})
            node.results["fidelities_nq"][qg.name] = fr.fidelity_nq
            node.results[f"{qg.name}_fidelity_nq"] = fr.fidelity_nq
            node.results["fidelity_differences"] = node.results.get("fidelity_differences", {})
            node.results["fidelity_differences"][qg.name] = fr.fidelity_difference

    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qg.name: ("successful" if fit_results[qg.name].success else "failed") for qg in node.namespace["qubit_groups"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot mitigated Z-basis population distributions."""
    qubit_groups = node.namespace["qubit_groups"]
    num_qubits = qubit_groups[0].num_qubits

    figures = plot_ghz_z_basis(
        {k: np.asarray(v) for k, v in node.results["corrected_results"].items()},
        qubit_groups,
        node.results["fidelities"],
        num_qubits=num_qubits,
        corrected_nq=(
            {k: np.asarray(v) for k, v in node.results["corrected_results_nq"].items()}
            if "corrected_results_nq" in node.results
            else None
        ),
        fidelities_nq=node.results.get("fidelities_nq"),
        fidelity_differences=node.results.get("fidelity_differences"),
    )
    node.results.update(figures)
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
