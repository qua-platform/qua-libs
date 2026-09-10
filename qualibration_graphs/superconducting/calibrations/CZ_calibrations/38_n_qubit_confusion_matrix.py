"""Multi-qubit readout confusion matrix calibration node."""

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils.qua_nested_sweeps import nested_sweep
from calibration_utils.n_qubit_confusion_matrix import (
    Parameters,
    compute_confusion_matrices,
    compute_kron_confusion_matrices,
    get_qubit_groups,
    is_confusion_matrix_valid,
    plot_confusion_matrices,
    save_confusion_to_qubit_pair_extras,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
This experiment measures readout error when simultaneously measuring N qubits (1 to 5).

Process:
1. Prepare all computational basis states (|00...0⟩ to |11...1⟩)
2. Perform simultaneous readout on all qubits
3. Build the confusion matrix from measurement results

Parameters:
- qubit_groups: dash-separated qubit names per group, e.g. ["qC4-qC3-qC2"]

Outcomes:
- N×N confusion matrix (N = 2^num_qubits) for each configured qubit group
- Kronecker-product reference matrices and direct-minus-Kron difference plots
- Single-qubit marginal confusion matrices (readout crosstalk from simultaneous measurement included)
- For groups with 3+ qubits, measured matrices are saved to qubit pair extras

Prerequisites:
- Calibrated single-qubit gates for all qubits in each group
- Calibrated readout for all qubits in each group
"""

node = QualibrationNode[Parameters, Quam](
    name="38_n_qubit_confusion_matrix",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes only."""
    node.parameters.qubit_groups = ["qD3-qD1-qD2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for N-qubit confusion matrix measurement."""
    node.namespace["qubit_groups"] = qubit_groups = get_qubit_groups(node)
    num_qubit_groups = len(qubit_groups)
    num_qubits = qubit_groups[0].num_qubits
    n_shots = node.parameters.num_shots

    sweep_axes = {
        # XarrayDataFetcher only accepts qubit/qubit_pair as the first axis in
        # qualibration-libs releases; values are still qubit group names.
        "qubit_pair": xr.DataArray([qg.name for qg in qubit_groups]),
        "n": xr.DataArray(
            np.arange(n_shots),
            attrs={"long_name": "shot index"},
        ),
    }
    for q_idx in range(num_qubits):
        sweep_axes[f"init_{q_idx}"] = xr.DataArray(
            [0, 1],
            attrs={"long_name": f"prepared qubit {q_idx} state"},
        )
    node.namespace["sweep_axes"] = sweep_axes

    with program() as node.namespace["qua_program"]:
        init_vars = [declare(int) for _ in range(num_qubits)]
        state_vars = [declare(int) for _ in range(num_qubits)]
        n = declare(int)
        n_st = declare_output_stream()
        state = [declare(int) for _ in range(num_qubit_groups)]
        state_st = [declare_output_stream() for _ in range(num_qubit_groups)]

        for group_idx, qg in enumerate(qubit_groups):
            for q in qg.qubits:
                node.machine.initialize_qpu(target=q)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)
                with nested_sweep(init_vars):
                    for q in qg.qubits:
                        q.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()

                    for idx, q in enumerate(qg.qubits):
                        with if_(init_vars[idx] == 1):
                            q.xy.play("x180")
                    align()

                    for idx, q in enumerate(qg.qubits):
                        q.readout_state(state_vars[idx])

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
                    state_stream = state_stream.buffer(2)
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
    node.namespace["qubit_groups"] = get_qubit_groups(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw data and compute confusion matrices."""
    qubit_groups = node.namespace["qubit_groups"]
    num_qubits = qubit_groups[0].num_qubits
    confusions = compute_confusion_matrices(
        node.results["ds_raw"],
        [qg.name for qg in qubit_groups],
        node.parameters.num_shots,
        [f"init_{idx}" for idx in range(num_qubits)],
        log_callable=node.log,
    )
    kron_confs = compute_kron_confusion_matrices({qg.name: qg.qubits for qg in qubit_groups})

    node.results["confusions"] = confusions
    node.results["kron_confs"] = kron_confs

    for qg in qubit_groups:
        conf = confusions[qg.name]
        node.results[f"{qg.name}_mean_assignment_fidelity"] = np.trace(conf) / conf.shape[0]

    node.outcomes = {
        qg.name: ("successful" if is_confusion_matrix_valid(confusions.get(qg.name, np.empty(0))) else "failed")
        for qg in qubit_groups
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot direct, Kronecker, and difference confusion matrices."""
    figures = plot_confusion_matrices(
        node.results["confusions"],
        node.results["kron_confs"],
        node.namespace["qubit_groups"],
        node=node,
    )
    for name, fig in figures.items():
        node.results[name] = fig
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Save measured confusion matrices to qubit pair extras."""
    qubit_groups = node.namespace["qubit_groups"]
    confusions = node.results["confusions"]
    with node.record_state_updates():
        for group in qubit_groups:
            if node.outcomes.get(group.name) != "successful":
                node.log(f"Skipping save for {group.name}: confusion matrix failed validation.")
                continue
            save_confusion_to_qubit_pair_extras(
                node.machine,
                [group],
                confusions,
                log_callable=node.log,
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
