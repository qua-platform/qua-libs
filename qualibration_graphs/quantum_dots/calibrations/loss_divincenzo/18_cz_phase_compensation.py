# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *
from qm.qua import fixed
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs, enable_dual_drive_mw_pairs
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import process_parity_streams
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.cz_phase_compensation import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        CZ PHASE COMPENSATION - using standard QUA (pulse > 16ns and 4ns granularity)
This program calibrates and compensates the residual single-qubit (local Z) phase shifts induced by the
geometric CZ gate. After calibrating the CZ operating point (node 16), the physical exchange interaction
imparts unconditional Z phases on both qubits in addition to the desired conditional pi phase. These
unconditional phases must be cancelled by virtual Z (frame) rotations applied after each CZ gate.

The measurement uses a Ramsey-like sequence: X90 — CZ(swept frame rotation) — X90, with parity readout.
Sweeping the virtual frame rotation over a full 2pi produces a sinusoidal signal whose fitted phase gives
the residual phase error. Four experiment types are run per qubit pair:
    Experiment 0: Target in superposition, control in |0⟩ → target unconditional phase (correction)
    Experiment 1: Target in superposition, control in |1⟩ → target + conditional phase (cross-check)
    Experiment 2: Control in superposition, target in |0⟩ → control unconditional phase (correction)
    Experiment 3: Control in superposition, target in |1⟩ → control + conditional phase (cross-check)

The ground-state-partner experiments (0, 2) give the phase corrections. The excited-state-partner
experiments (1, 3) verify that the conditional phase is pi.

Prerequisites:
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having calibrated the CZ gate operating point (node 16): conditional phase ~pi, SWAP ~2pi.

State update:
    - CZ macro phase_shift_control
    - CZ macro phase_shift_target
"""


node = QualibrationNode[Parameters, Quam](
    name="18_cz_phase_compensation",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 2
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    frames = np.linspace(0, 1, node.parameters.num_frames, endpoint=False)

    experiment_type_values = [0, 1, 2, 3]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "experiment_type": xr.DataArray(
            experiment_type_values,
            attrs={
                "long_name": "experiment type",
                "units": "",
                "description": "0=target_ctrl0, 1=target_ctrl1, 2=control_tgt0, 3=control_tgt1",
            },
        ),
        "frame": xr.DataArray(
            frames,
            attrs={"long_name": "frame rotation", "units": "2π"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        frame = declare(fixed)

        p2 = declare(int)
        p_st = {qp.name: declare_output_stream() for qp in qubit_pairs}

        stored_phase_control = declare(fixed)
        stored_phase_target = declare(fixed)

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                assign(stored_phase_control, qubit_pair.macros["cz"].phase_shift_control)
                assign(stored_phase_target, qubit_pair.macros["cz"].phase_shift_target)

                for exp_type_val in experiment_type_values:
                    with for_(*from_array(frame, frames)):

                        reset_frame(
                            qubit_pair.qubit_target.xy.name,
                            qubit_pair.qubit_control.xy.name
                        )

                        qubit_pair.initialize()

                        align()

                        # State preparation
                        if exp_type_val == 0:
                            qubit_pair.qubit_target.x90()
                        elif exp_type_val == 1:
                            qubit_pair.qubit_control.x180()
                            qubit_pair.qubit_target.x90()
                        elif exp_type_val == 2:
                            qubit_pair.qubit_control.x90()
                        elif exp_type_val == 3:
                            qubit_pair.qubit_target.x180()
                            qubit_pair.qubit_control.x90()

                        align()

                        # CZ gate with swept frame rotation on the qubit under test
                        if exp_type_val in (0, 1):
                            qubit_pair.cz(
                                phase_shift_target= frame + stored_phase_target
                            )
                        else:
                            qubit_pair.cz(
                                phase_shift_control= frame + stored_phase_control
                            )

                        align()

                        # Closing x90 on qubit under test
                        if exp_type_val in (0, 1):
                            qubit_pair.qubit_target.x90()
                        else:
                            qubit_pair.qubit_control.x90()

                        # Undo the partner spin flip so PSB parity
                        # readout maps to the qubit under test
                        if exp_type_val == 1:
                            qubit_pair.qubit_control.x180()
                        elif exp_type_val == 3:
                            qubit_pair.qubit_target.x180()

                        align()

                        a2 = qubit_pair.measure()

                        align()

                        qubit_pair.cz.balance()

                        align()

                        qubit_pair.voltage_sequence.ramp_to_zero()
                        align()

                        assign(p2, Cast.to_int(a2))
                        save(p2, p_st[qubit_pair.name])

        with stream_processing():
            n_st.save("n")

            n_exp_types = len(experiment_type_values)
            n_frames = len(frames)
            for qubit_pair in qubit_pairs:
                p_st[qubit_pair.name].buffer(n_exp_types, n_frames).average().save(
                    f"p_{qubit_pair.name}"
                )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [qp.name for qp in node.namespace["qubit_pairs"]],
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("experiment_type", "frame"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit sinusoids, extract phase corrections and conditional phase cross-checks."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        analysis_signal=node.parameters.analysis_signal,
        conditional_phase_tolerance=node.parameters.conditional_phase_tolerance,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        name: ("successful" if r["success"] else "failed")
        for name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot raw data and fitted sinusoids for all qubit pairs."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the CZ macro phase corrections for successful pairs."""

    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            if not node.results["fit_results"][qubit_pair.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit_pair.name]

            old_phase_control = qubit_pair.macros["cz"].phase_shift_control
            old_phase_target = qubit_pair.macros["cz"].phase_shift_target

            qubit_pair.macros["cz"].phase_shift_control = (
                old_phase_control - fit_result["control_phase_correction"]
            ) % 1
            qubit_pair.macros["cz"].phase_shift_target = (
                old_phase_target - fit_result["target_phase_correction"]
            ) % 1


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
