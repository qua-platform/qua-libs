"""GHZ state tomography calibration node."""

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

from calibration_utils.common_utils.qua_nested_sweeps import nested_sweep
from calibration_utils.n_qubit_confusion_matrix import (
    get_qubit_groups,
    require_adjacent_cz_macros,
)
from calibration_utils.ghz_tomography import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_ghz_tomography,
)

# %% {Initialisation}
description = """
**MULTI-QUBIT GHZ STATE TOMOGRAPHY**

This experiment prepares an N-qubit GHZ state (N >= 2) and performs full tomography by
sweeping local X/Y/Z pre-rotation axes on each qubit.

**Topology:** GHZ preparation is hardcoded as a linear chain (ladder circuit): CZ on
consecutive qubits ``pair_01``, ``pair_12``, … in list order. This is not a general
graph/topology builder. Order ``qubit_groups`` so each consecutive pair exists in the
machine (e.g. star couplers ``qD2–qD1`` and ``qD3–qD1`` require
``["qD2", "qD1", "qD3"]``, hub in the middle).

Analysis applies readout mitigation in two ways:

- **kron**: tensor product of per-qubit resonator confusion matrices
- **nq**: full N-qubit confusion matrix from node 38_n_qubit_confusion_matrix

For each method the node reconstructs the density matrix and reports fidelity and purity
relative to the ideal GHZ target state.

Prerequisites:
- Calibrated single-qubit control and readout for all qubits in each chain
- Available nearest-neighbor CZ operations along the chain
- Valid readout confusion matrices for mitigation (kron and/or nq)
"""

node = QualibrationNode[Parameters, Quam](
    name="39b_ghz_tomography",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes only."""
    node.parameters.qubit_groups = [["qD3", "qD1", "qD2"]]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for GHZ tomography."""
    node.namespace["qubit_groups"] = qubit_groups = get_qubit_groups(node, min_qubits=2, resolve_adjacent_pairs=True)
    operation = node.parameters.operation
    require_adjacent_cz_macros(qubit_groups, operation)

    num_qubit_groups = len(qubit_groups)
    num_qubits = qubit_groups[0].num_qubits
    n_shots = node.parameters.num_shots

    sweep_axes = {
        "qubit_pair": xr.DataArray([qg.name for qg in qubit_groups]),
        "n": xr.DataArray(
            np.arange(n_shots),
            attrs={"long_name": "shot index"},
        ),
    }
    for idx in range(num_qubits):
        sweep_axes[f"tomo_axis_{idx}"] = xr.DataArray(
            [0, 1, 2],
            attrs={"long_name": f"tomography axis qubit {idx} (0=X, 1=Y, 2=Z)"},
        )
    node.namespace["sweep_axes"] = sweep_axes

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        state_vars = [declare(int) for _ in range(num_qubits)]
        state_st_vars = [declare_output_stream() for _ in range(num_qubits)]
        state = [declare(int) for _ in range(num_qubit_groups)]
        state_st = [declare_output_stream() for _ in range(num_qubit_groups)]
        tomo_axes = [declare(int) for _ in range(num_qubits)]

        for group_idx, qg in enumerate(qubit_groups):
            for q in qg.qubits:
                node.machine.initialize_qpu(target=q)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)
                with nested_sweep(tomo_axes, upper=3):
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

                    for idx, q in enumerate(qg.qubits):
                        with if_(tomo_axes[idx] == 0):
                            q.xy.play("y90")
                        with if_(tomo_axes[idx] == 1):
                            q.xy.play("x90")
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
                state_stream = state_st[group_idx]
                for _ in range(num_qubits):
                    state_stream = state_stream.buffer(3)
                state_stream.buffer(n_shots).save(f"state{group_idx + 1}")


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
    node.namespace["qubit_groups"] = get_qubit_groups(node, min_qubits=2, resolve_adjacent_pairs=True)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Reconstruct density matrices and compute GHZ fidelity and purity."""
    rhos_by_method, paulis_by_method, fit_results = fit_raw_data(node.results["ds_raw"], node)

    node.results["rhos"] = rhos_by_method
    node.results["paulis_data"] = paulis_by_method
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    for qg in node.namespace["qubit_groups"]:
        fr = fit_results[qg.name]
        node.results[f"{qg.name}_fidelity_kron"] = fr.fidelity_kron
        node.results[f"{qg.name}_purity_kron"] = fr.purity_kron
        node.results[f"{qg.name}_fidelity"] = fr.fidelity_kron
        node.results[f"{qg.name}_purity"] = fr.purity_kron
        if fr.fidelity_nq is not None and fr.purity_nq is not None:
            node.results[f"{qg.name}_fidelity_nq"] = fr.fidelity_nq
            node.results[f"{qg.name}_purity_nq"] = fr.purity_nq

    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qg.name: ("successful" if fit_results[qg.name].success else "failed") for qg in node.namespace["qubit_groups"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot reconstructed density matrices."""
    qubit_groups = node.namespace["qubit_groups"]
    figures = plot_ghz_tomography(
        node.results["rhos"],
        qubit_groups,
        node.results["fit_results"],
        num_qubits=qubit_groups[0].num_qubits,
        plot_level=node.parameters.plot_level,
    )
    for name, fig in figures.items():
        node.results[name] = fig
    plt.show()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
