"""Two-qubit readout confusion matrix calibration node."""

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from calibration_utils.two_q_confusion_matrix import (
    Parameters,
    compute_confusion_matrices,
    is_confusion_matrix_valid,
    plot_confusion_matrices,
    process_raw_dataset,
)

# %% {Initialisation}
description = """
**TWO-QUBIT READOUT CONFUSION MATRIX MEASUREMENT**

This experiment measures the readout error when simultaneously measuring the state of two qubits.

The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure the readout result of both qubits. The measurement process involves:
initializing both qubits to the ground state, applying single-qubit gates to prepare the desired input state,
and performing simultaneous readout on both qubits.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

node = QualibrationNode[Parameters, Quam](
    name="35_two_qubit_confusion_matrix",
    description=description,
    parameters=Parameters(),
    machine = Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes only."""
    # node.parameters.qubit_pairs = ["q1_q2"]
    pass

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for confusion matrix measurement."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    n_shots = node.parameters.num_shots

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "n": xr.DataArray(
            np.arange(n_shots),
            attrs={"long_name": "shot index"},
        ),
        "init_state_control": xr.DataArray(
            [0, 1],
            attrs={"long_name": "prepared control qubit state"},
        ),
        "init_state_target": xr.DataArray(
            [0, 1],
            attrs={"long_name": "prepared target qubit state"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        control_initial = declare(int)
        target_initial = declare(int)
        n = declare(int)
        n_st = declare_output_stream()
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(control_initial, [0, 1])):
                    with for_(*from_array(target_initial, [0, 1])):
                        for qp in multiplexed_qubit_pairs.values():
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        for qp in multiplexed_qubit_pairs.values():
                            with if_(control_initial == 1):
                                qp.qubit_control.xy.play("x180")
                            with if_(target_initial == 1):
                                qp.qubit_target.xy.play("x180")
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.readout_state(state_control[ii])
                            qp.qubit_target.readout_state(state_target[ii])
                            assign(state[ii], state_control[ii] * 2 + state_target[ii])
                            save(state_control[ii], state_st_control[ii])
                            save(state_target[ii], state_st_target[ii])
                            save(state[ii], state_st[ii])
                        align()
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_st_control[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_control{i + 1}")
                state_st_target[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_target{i + 1}")
                state_st[i].buffer(2).buffer(2).buffer(n_shots).save(f"state{i + 1}")


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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw data and compute confusion matrices."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    confusions = compute_confusion_matrices(node.results["ds_raw"], node, log_callable=node.log)
    node.results["confusions"] = confusions
    node.outcomes = {
        qp.name: ("successful" if is_confusion_matrix_valid(confusions.get(qp.name, np.empty(0))) else "failed")
        for qp in node.namespace["qubit_pairs"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot confusion matrix results."""
    qubit_pairs = node.namespace["qubit_pairs"]
    figures = plot_confusion_matrices(
        node.results["confusions"],
        qubit_pairs,
        node,
    )
    for name, fig in figures.items():
        node.results[name] = fig
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the qubit pair state with confusion matrices."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes.get(qp.name) != "successful":
                continue
            node.machine.qubit_pairs[qp.id].confusion = node.results["confusions"][qp.name].tolist()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
