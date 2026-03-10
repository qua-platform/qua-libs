# %% {Imports}
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_virtualization import (
    BarrierCompensationParameters,
    plot_2d_scan,
    plot_barrier_pair_diagnostics,
    plot_target_barrier_coupling_summary,
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
    evaluate_slope_fit_acceptance,
    extract_barrier_compensation_coefficients,
    resolve_pair_calibration_topology,
)

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        BARRIER-BARRIER VIRTUALIZATION — PAIR-DRIVEN STEPWISE B* -> B†
Implements pair-driven barrier virtualization consistent with the stepwise
supplementary method in E. Hsiao et al. (arXiv:2001.07671).

Input is an ordered list of target pair names (`pair_names`), resolved from
machine.quantum_dot_pairs or machine.qubit_pairs. For each target pair i:
- target barrier is taken from pair.barrier_gate
- nearest-neighbor drive barriers are discovered from all quantum-dot pairs
  sharing exactly one dot with the target pair
- tunnel coupling is extracted from detuning sweeps at each drive setpoint
- slope matrix row m_ij = dt_i/dB_j is built and quality-gated

Rows are composed stepwise into B* snapshots and final B†, with enforced
unit diagonal self terms r_ii = 1.
"""


node = QualibrationNode[BarrierCompensationParameters, Quam](
    name="03_barrier_compensation",
    description=description,
    parameters=BarrierCompensationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Allow local debug parameter overrides."""
    # node.parameters.pair_names = ["qd_pair_1_2", "qd_pair_2_3", "qd_pair_3_4"]
    pass


node.machine = Quam.load()


def _get_active_virtual_gate_set_id(node: QualibrationNode[BarrierCompensationParameters, Quam]) -> str:
    vgs_id = getattr(node.parameters, "virtual_gate_set_id", None)
    if vgs_id:
        return str(vgs_id)

    virtual_gate_sets = getattr(node.machine, "virtual_gate_sets", {})
    if len(virtual_gate_sets) == 1:
        return str(next(iter(virtual_gate_sets.keys())))

    raise ValueError("virtual_gate_set_id must be set when multiple VirtualGateSets exist.")


def _resolve_target_lever_arm(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
    target_barrier: str,
    dot_pair_id: str,
) -> float:
    """Resolve detuning scaling factor for a target barrier.

    Priority:
    1) barrier_lever_arm_mapping[target_barrier]
    2) pat_lever_arm_mapping[dot_pair_id]
    3) default_lever_arm
    """
    barrier_map = node.parameters.barrier_lever_arm_mapping or {}
    if target_barrier in barrier_map:
        return float(barrier_map[target_barrier])

    pat_map = node.parameters.pat_lever_arm_mapping or {}
    if dot_pair_id in pat_map:
        return float(pat_map[dot_pair_id])

    return float(node.parameters.default_lever_arm)


def _resolve_barrier_center(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
    barrier_name: str,
    active_vgs_id: str,
) -> float:
    """Resolve barrier center from override, virtual-DC tracked level, or zero."""
    overrides = node.parameters.barrier_center_overrides or {}
    if barrier_name in overrides:
        return float(overrides[barrier_name])

    virtual_dc_sets = getattr(node.machine, "virtual_dc_sets", {})
    virtual_dc = virtual_dc_sets.get(active_vgs_id)
    if virtual_dc is not None:
        current_levels = getattr(virtual_dc, "_current_levels", None)
        if isinstance(current_levels, dict) and barrier_name in current_levels:
            try:
                return float(current_levels[barrier_name])
            except Exception:
                pass

    return 0.0


def _resolve_detuning_center(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
    target_pair_id: str,
) -> float:
    """Resolve detuning center from override or zero."""
    overrides = node.parameters.detuning_center_overrides or {}
    return float(overrides.get(target_pair_id, 0.0))


def _build_drive_and_detuning_arrays(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
    drive_center: float,
    detuning_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build local sweep arrays around configured centers."""
    drive_span_v = float(node.parameters.slope_sweep_span_mv) * 1e-3
    drive_values = drive_center + np.linspace(
        -0.5 * drive_span_v,
        0.5 * drive_span_v,
        int(node.parameters.slope_sweep_points),
    )
    detuning_values = detuning_center + np.linspace(
        float(node.parameters.detuning_min),
        float(node.parameters.detuning_max),
        int(node.parameters.detuning_points),
    )
    return drive_values, detuning_values


def _create_pair_detuning_program(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
    target_pair,
    target_barrier: str,
    drive_barrier: str,
    static_barrier_centers: dict[str, float],
    drive_values: np.ndarray,
    detuning_values: np.ndarray,
):
    """Create one QUA program for a target-pair detuning campaign."""
    sensor_dots = list(getattr(target_pair, "sensor_dots", []) or [])
    if len(sensor_dots) == 0:
        raise ValueError(f"Target pair '{target_pair.id}' has no sensor_dots for readout.")

    sensor = sensor_dots[0]
    rr = sensor.readout_resonator
    seq = target_pair.voltage_sequence

    has_empty = callable(getattr(target_pair, "empty", None))
    has_initialize = callable(getattr(target_pair, "initialize", None))

    static_bias = {name: float(value) for name, value in static_barrier_centers.items() if name != drive_barrier}

    with program() as qua_prog:
        n = declare(int)
        n_st = declare_stream()

        drive = declare(fixed)
        detuning = declare(fixed)

        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            with for_(*from_array(drive, drive_values)):
                # Keep all non-driven relevant barriers at center and set driven barrier.
                seq.ramp_to_voltages(
                    {**static_bias, drive_barrier: drive},
                    duration=node.parameters.hold_duration,
                    ramp_duration=node.parameters.ramp_duration,
                )

                with for_(*from_array(detuning, detuning_values)):
                    if has_empty:
                        target_pair.empty()
                    if has_initialize:
                        target_pair.initialize()

                    target_pair.ramp_to_detuning(
                        detuning,
                        ramp_duration=node.parameters.ramp_duration,
                        duration=node.parameters.hold_duration,
                    )

                    if node.parameters.pre_measurement_delay > 0:
                        seq.step_to_voltages({}, duration=node.parameters.pre_measurement_delay)

                    align()
                    rr.measure("readout", qua_vars=(I, Q))
                    rr.wait(500)
                    save(I, I_st)
                    save(Q, Q_st)

                if node.parameters.per_line_compensation:
                    seq.apply_compensation_pulse()

            seq.apply_compensation_pulse()

        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(detuning_values)).buffer(len(drive_values)).average().save("I")
            Q_st.buffer(len(detuning_values)).buffer(len(drive_values)).average().save("Q")

    sweep_axes = {
        "drive_volts": xr.DataArray(
            drive_values,
            attrs={"long_name": f"{drive_barrier} voltage", "units": "V"},
        ),
        "detuning_volts": xr.DataArray(
            detuning_values,
            attrs={"long_name": f"{target_pair.detuning_axis_name} voltage", "units": "V"},
        ),
    }
    return qua_prog, sweep_axes


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Create pair-driven barrier-vs-detuning scan programs."""
    pair_names = node.parameters.pair_names or []
    topology = resolve_pair_calibration_topology(node.machine, pair_names)

    active_vgs_id = _get_active_virtual_gate_set_id(node)

    programs = {}
    sweep_axes_all = {}
    pair_metadata = {}

    for resolved in topology["pair_resolution"]:
        target_pair_id = str(resolved["target_pair_id"])
        target_pair = node.machine.quantum_dot_pairs[target_pair_id]
        target_barrier = str(resolved["target_barrier"])

        target_gate_set = None
        try:
            target_gate_set = target_pair.voltage_sequence.gate_set.id
        except Exception:
            pass
        if target_gate_set is not None and str(target_gate_set) != active_vgs_id:
            raise ValueError(
                f"Pair '{target_pair_id}' belongs to VirtualGateSet '{target_gate_set}', "
                f"but active node set is '{active_vgs_id}'."
            )

        detuning_axis_name = str(resolved["detuning_axis_name"])
        if not callable(getattr(target_pair, "ramp_to_detuning", None)):
            raise ValueError(f"Pair '{target_pair_id}' does not support ramp_to_detuning.")

        for drive_barrier in list(resolved["drive_barriers"]):
            drive_center = _resolve_barrier_center(node, drive_barrier, active_vgs_id)
            detuning_center = _resolve_detuning_center(node, target_pair_id)
            drive_values, detuning_values = _build_drive_and_detuning_arrays(
                node,
                drive_center=drive_center,
                detuning_center=detuning_center,
            )

            static_centers = {
                barrier_name: _resolve_barrier_center(node, barrier_name, active_vgs_id)
                for barrier_name in list(resolved["drive_barriers"])
            }

            pair_key = f"{target_pair_id}__{target_barrier}_vs_{drive_barrier}"
            qua_prog, sweep_axes = _create_pair_detuning_program(
                node=node,
                target_pair=target_pair,
                target_barrier=target_barrier,
                drive_barrier=drive_barrier,
                static_barrier_centers=static_centers,
                drive_values=drive_values,
                detuning_values=detuning_values,
            )

            programs[pair_key] = qua_prog
            sweep_axes_all[pair_key] = sweep_axes
            pair_metadata[pair_key] = {
                **resolved,
                "target_pair_id": target_pair_id,
                "target_barrier": target_barrier,
                "drive_barrier": str(drive_barrier),
                "drive_axis": "drive_volts",
                "detuning_axis": "detuning_volts",
                "detuning_axis_name": detuning_axis_name,
                "drive_center": float(drive_center),
                "detuning_center": float(detuning_center),
            }

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all
    node.namespace["pair_metadata"] = pair_metadata
    node.namespace["pair_resolution"] = topology["pair_resolution"]
    node.namespace["barrier_order"] = topology["barrier_order"]
    node.namespace["drive_mapping"] = topology["drive_mapping"]
    node.namespace["target_barrier_to_pair_id"] = topology["target_barrier_to_pair_id"]


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

    pair_metadata = node.namespace.get("pair_metadata", {})
    barrier_order = list(node.namespace.get("barrier_order", []))
    if len(barrier_order) == 0:
        raise ValueError("Missing barrier_order in node namespace.")

    ds_processed_all = {}
    fit_results = {}
    used_lever_arms = {}
    sensitivity_report = {}

    for pair_key, ds_raw in ds_raw_all.items():
        if pair_key not in pair_metadata:
            raise KeyError(f"Missing pair metadata for dataset key '{pair_key}'.")

        meta = dict(pair_metadata[pair_key])
        target_barrier = str(meta["target_barrier"])
        drive_barrier = str(meta["drive_barrier"])
        target_pair_id = str(meta["target_pair_id"])

        detuning_scale = _resolve_target_lever_arm(
            node,
            target_barrier=target_barrier,
            dot_pair_id=target_pair_id,
        )

        ds = process_raw_dataset(ds_raw, node)
        fit = extract_barrier_compensation_coefficients(
            ds,
            barrier_gate_name=str(meta.get("drive_axis", "drive_volts")),
            compensation_gate_name=str(meta.get("detuning_axis", "detuning_volts")),
            detuning_scale=detuning_scale,
        )
        quality = evaluate_slope_fit_acceptance(
            fit,
            min_slope_snr=float(node.parameters.min_slope_snr),
            min_tunnel_span_sigma=float(node.parameters.min_tunnel_span_sigma),
            min_pair_fit_r2=float(node.parameters.min_pair_fit_r2),
        )

        fit.update(meta)
        fit["detuning_scale"] = float(detuning_scale)
        fit["accepted"] = bool(quality["accepted"])
        fit["sensitivity"] = quality

        # Only accepted fits contribute to slope matrix construction.
        if not quality["accepted"]:
            fit["success"] = False
            fit["coefficient"] = float("nan")

        ds_processed_all[pair_key] = ds
        fit_results[pair_key] = fit
        used_lever_arms[pair_key] = float(detuning_scale)
        sensitivity_report[pair_key] = quality

    # Enforce valid self-terms for every target row before matrix construction.
    for target_barrier in barrier_order:
        self_fit = None
        for pair_key, fit in fit_results.items():
            if fit.get("target_barrier") == target_barrier and fit.get("drive_barrier") == target_barrier:
                self_fit = (pair_key, fit)
                break

        if self_fit is None:
            raise ValueError(f"Missing self-slope fit for target barrier '{target_barrier}'.")

        pair_key, fit = self_fit
        if not bool(fit.get("accepted", False)):
            reasons = fit.get("sensitivity", {}).get("reasons", [])
            raise ValueError(
                f"Self-slope fit for target barrier '{target_barrier}' failed acceptance "
                f"for pair '{pair_key}'. Reasons: {reasons}"
            )

    slope_matrix_raw = assemble_slope_matrix(fit_results, barrier_order)
    if np.any(np.isnan(np.diag(slope_matrix_raw))):
        raise ValueError(
            "Missing self-slope rows in slope_matrix_raw. " "Ensure each target barrier has an accepted self fit."
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

    node.results["pair_resolution"] = node.namespace.get("pair_resolution", [])
    node.results["drive_mapping"] = node.namespace.get("drive_mapping", {})
    node.results["ds_processed_all"] = ds_processed_all
    node.results["fit_results"] = fit_results
    node.results["sensitivity_report"] = sensitivity_report
    node.results["slope_matrix_raw"] = stepwise["slope_matrix_raw"]
    node.results["barrier_transform_history"] = stepwise["barrier_transform_history"]
    node.results["barrier_transform_final"] = stepwise["barrier_transform_final"]
    node.results["residual_crosstalk"] = stepwise["residual_crosstalk"]
    node.results["max_residual_crosstalk"] = stepwise["max_residual_crosstalk"]
    node.results["effective_slope_matrix"] = stepwise["effective_slope_matrix"]
    node.results["barrier_order"] = barrier_order
    node.results["lever_arms_used"] = used_lever_arms

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

        x_axis = fit.get("barrier_axis", fit.get("drive_axis", "drive_volts"))
        y_axis = fit.get("compensation_axis", fit.get("detuning_axis", "detuning_volts"))

        figures[f"scan_{pair_key}"] = plot_2d_scan(
            ds,
            x_axis=x_axis,
            y_axis=y_axis,
            title=f"Barrier Scan: {target} vs {drive}",
        )
        figures[f"fit_{pair_key}"] = plot_barrier_pair_diagnostics(
            fit,
            drive_label=drive,
            target_label=str(target),
            title=f"Slope Extraction: {target} vs {drive}",
        )

    # Per-target summary: t(target) vs each drive barrier with linear fits.
    by_target: dict[str, dict[str, dict]] = {}
    for pair_key, fit in fit_results.items():
        target = str(fit.get("target_barrier", ""))
        if target == "":
            continue
        by_target.setdefault(target, {})[pair_key] = fit
    for target, target_fit_map in by_target.items():
        figures[f"target_summary_{target}"] = plot_target_barrier_coupling_summary(
            target_barrier=target,
            fit_results_by_pair=target_fit_map,
            title=f"{target}: tunnel coupling vs barrier voltage",
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
