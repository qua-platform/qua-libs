# %% {Imports}
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_virtualization import (
    BarrierCompensationParameters,
    create_2d_scan_program,
    plot_2d_scan,
    plot_compensation_fit,
    plot_barrier_transform_history,
    plot_virtual_gate_matrix,
)
from calibration_utils.gate_virtualization.analysis import (
    process_raw_dataset,
    update_compensation_submatrix,
)
from calibration_utils.gate_virtualization.barrier_compensation_analysis import (
    assemble_slope_matrix,
    calibrate_stepwise_barrier_virtualization,
    extract_barrier_compensation_coefficients,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        BARRIER-BARRIER VIRTUALIZATION — STEPWISE B* -> B†
Implements an offline-first version of the barrier virtualization method from:
E. Hsiao et al., arXiv:2001.07671.

For each target barrier, the node estimates local derivatives dt_i/dB_j from
small barrier sweeps, builds normalized row coefficients, and composes a
stepwise transform matrix (B*1, B*2, ..., B†) to suppress off-diagonal
barrier crosstalk.

In this first implementation, data acquisition remains based on 2D scans and
analysis is robust to synthetic/offline datasets.

Prerequisites:
    - Calibrated sensor gate compensation (node 01).
    - Calibrated virtual plunger gates (node 02).
    - Barrier-compensation mapping including self terms for each target barrier.
    - Configured VirtualGateSet with a square layer including barrier gates.
"""


node = QualibrationNode[BarrierCompensationParameters, Quam](
    name="03_barrier_compensation",
    description=description,
    parameters=BarrierCompensationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Allow local debug parameter overrides."""
    # node.parameters.barrier_compensation_mapping = {
    #     "barrier_12": ["barrier_12", "barrier_23"],
    #     "barrier_23": ["barrier_12", "barrier_23", "barrier_34"],
    #     "barrier_34": ["barrier_23", "barrier_34"],
    # }
    pass


node.machine = Quam.load()


def _ensure_self_included(target_barrier: str, drive_barriers: list[str]) -> list[str]:
    """Ensure the self drive term is present and first in each row mapping."""
    ordered = [target_barrier]
    ordered.extend(g for g in drive_barriers if g != target_barrier)
    return ordered


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Create 2D scan programs for each target-vs-drive barrier pair."""
    node.namespace["sensors"] = sensors = get_sensors(node)

    mapping = node.parameters.barrier_compensation_mapping
    if mapping is None or len(mapping) == 0:
        raise ValueError("barrier_compensation_mapping must be provided and non-empty.")

    barrier_dot_pair_mapping = node.parameters.barrier_dot_pair_mapping or {}
    missing_dot_pairs = [name for name in mapping if name not in barrier_dot_pair_mapping]
    if missing_dot_pairs:
        node.log(
            "barrier_dot_pair_mapping missing keys " f"{missing_dot_pairs}; proceeding in offline-compatible mode."
        )

    programs = {}
    sweep_axes_all = {}
    pair_metadata = {}

    for target_barrier, drives in mapping.items():
        drive_barriers = _ensure_self_included(target_barrier, list(drives))
        for drive_barrier in drive_barriers:
            pair_key = f"{target_barrier}_vs_{drive_barrier}"
            # In current scan builder we use X for drive and Y as target/proxy detuning axis.
            node.parameters.x_axis_name = drive_barrier
            node.parameters.y_axis_name = target_barrier
            qua_prog, sweep_axes = create_2d_scan_program(node, sensors)
            programs[pair_key] = qua_prog
            sweep_axes_all[pair_key] = sweep_axes
            pair_metadata[pair_key] = {
                "target_barrier": target_barrier,
                "drive_barrier": drive_barrier,
                "dot_pair": barrier_dot_pair_mapping.get(target_barrier),
            }

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all
    node.namespace["pair_metadata"] = pair_metadata


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
    """Simulate the first generated scan program for quick sanity checks."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    first_key = next(iter(node.namespace["programs"]))
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["programs"][first_key], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.run_in_video_mode
)
def execute_qua_program(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
    """Execute all generated pair scans and store raw datasets."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    datasets = {}
    for pair_key, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes_all"][pair_key])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            node.log(f"[{pair_key}] {job.execution_report()}")
        datasets[pair_key] = dataset

    node.results["ds_raw_all"] = datasets


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Load previously acquired data (placeholder)."""
    node.log("Historical loading for node 03 is not yet implemented.")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Run barrier-barrier analysis and build the stepwise transform."""
    ds_raw_all = node.results.get("ds_raw_all", {})
    if not ds_raw_all:
        raise ValueError("No raw datasets available. Expected node.results['ds_raw_all'].")

    mapping = node.parameters.barrier_compensation_mapping or {}
    barrier_order = list(mapping.keys())
    if not barrier_order:
        raise ValueError("barrier_compensation_mapping must be set for analysis.")

    pair_metadata = node.namespace.get("pair_metadata", {})

    ds_processed_all = {}
    fit_results = {}
    for pair_key, ds_raw in ds_raw_all.items():
        meta = pair_metadata.get(pair_key, {})
        if "target_barrier" not in meta or "drive_barrier" not in meta:
            if "_vs_" in pair_key:
                target, drive = pair_key.split("_vs_", 1)
            else:
                raise ValueError(f"Cannot infer pair metadata from key '{pair_key}'.")
            meta = {
                "target_barrier": target,
                "drive_barrier": drive,
                "dot_pair": None,
            }

        ds = process_raw_dataset(ds_raw, node)
        fit = extract_barrier_compensation_coefficients(
            ds,
            barrier_gate_name=meta["drive_barrier"],
            compensation_gate_name=meta["target_barrier"],
        )
        fit.update(meta)

        ds_processed_all[pair_key] = ds
        fit_results[pair_key] = fit

    slope_matrix_raw = assemble_slope_matrix(fit_results, barrier_order)
    if np.any(np.isnan(np.diag(slope_matrix_raw))):
        raise ValueError(
            "Missing self-slope rows in slope_matrix_raw. "
            "Ensure each target barrier includes itself in the drive list."
        )

    calibration_order = node.parameters.calibration_order or barrier_order
    stepwise = calibrate_stepwise_barrier_virtualization(
        slope_matrix_raw=slope_matrix_raw,
        barrier_names=barrier_order,
        calibration_order=calibration_order,
        residual_target=node.parameters.residual_crosstalk_target,
        max_refinement_rounds=node.parameters.max_refinement_rounds,
        min_abs_self_slope=node.parameters.min_abs_self_slope,
    )

    node.results["ds_processed_all"] = ds_processed_all
    node.results["fit_results"] = fit_results
    node.results["slope_matrix_raw"] = stepwise["slope_matrix_raw"]
    node.results["barrier_transform_history"] = stepwise["barrier_transform_history"]
    node.results["barrier_transform_final"] = stepwise["barrier_transform_final"]
    node.results["residual_crosstalk"] = stepwise["residual_crosstalk"]
    node.results["max_residual_crosstalk"] = stepwise["max_residual_crosstalk"]
    node.results["effective_slope_matrix"] = stepwise["effective_slope_matrix"]
    node.results["barrier_order"] = barrier_order

    if node.parameters.target_tunnel_couplings is not None:
        node.results["target_tunnel_couplings"] = dict(node.parameters.target_tunnel_couplings)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Create plots for pair scans, slope fits, and matrix evolution."""
    figures = {}

    ds_processed_all = node.results.get("ds_processed_all", {})
    fit_results = node.results.get("fit_results", {})

    for pair_key, ds in ds_processed_all.items():
        fit = fit_results.get(pair_key, {})
        target = fit.get("target_barrier", "target")
        drive = fit.get("drive_barrier", "drive")

        figures[f"scan_{pair_key}"] = plot_2d_scan(
            ds,
            title=f"Barrier Scan: {target} vs {drive}",
        )
        figures[f"fit_{pair_key}"] = plot_compensation_fit(
            ds,
            fit,
            gate_x_name=drive,
            gate_y_name=target,
            title=f"Slope Extraction: {target} vs {drive}",
        )

    barrier_order = node.results.get("barrier_order", [])
    if "barrier_transform_history" in node.results:
        figures["barrier_transform_history"] = plot_barrier_transform_history(
            node.results["barrier_transform_history"],
            barrier_order,
            title="Barrier Transform Evolution (B* -> B†)",
        )

    if "barrier_transform_final" in node.results:
        figures["barrier_transform_final"] = plot_virtual_gate_matrix(
            np.asarray(node.results["barrier_transform_final"], dtype=float),
            list(barrier_order),
            title="Final Barrier Transform (B†)",
        )

    node.results["figures"] = figures
    if figures:
        node.results["figure"] = figures.get("barrier_transform_final") or next(iter(figures.values()))


# %% {Update_virtual_gate_matrix}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def update_virtual_gate_matrix(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
    """Update the active virtual-gate layer with the final barrier transform."""
    if "barrier_transform_final" not in node.results:
        node.log("No barrier_transform_final found; skipping matrix update.")
        return

    barrier_order = node.results.get("barrier_order")
    if not barrier_order:
        raise ValueError("Missing barrier_order in node.results.")

    transform = np.asarray(node.results["barrier_transform_final"], dtype=float)

    context = node.record_state_updates() if hasattr(node, "record_state_updates") else nullcontext()
    with context:
        update_meta = update_compensation_submatrix(
            node=node,
            row_names=barrier_order,
            col_names=barrier_order,
            values=transform,
            layer_id=node.parameters.matrix_layer_id,
        )

    node.results["matrix_update"] = update_meta


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Persist node outputs and any state updates."""
    node.save()
