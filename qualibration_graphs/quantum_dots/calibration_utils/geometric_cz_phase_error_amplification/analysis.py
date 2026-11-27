"""Analysis for the 2-D CZ phase error-amplification calibration.

Two independent Ramsey-like sweeps are performed:

  A) **Target qubit phase sweep**
       X90(target) – N×CZ – θ-frame-rotation(target) – X90(target) – measure(target)
       Conditional state: control qubit prepared in |0⟩ or |1⟩.
       D_T(N,θ) = S_T(ctrl=|1⟩) − S_T(ctrl=|0⟩)

  B) **Control qubit phase sweep**
       X90(control) – N×CZ – θ-frame-rotation(control) – X90(control) – measure(control)
       Conditional state: target qubit prepared in |0⟩ or |1⟩.
       D_C(N,θ) = S_C(tgt=|1⟩) − S_C(tgt=|0⟩)

Both differences take the form:

    D(N,θ) ≈ A · cos(N·δφ/2) · cos(θ − φ_acc)

where φ_acc is the single-axis phase accumulated per CZ gate by the respective
qubit and δφ = J·t/2 − π/2 is the amplitude error.

The mean absolute difference ⟨|D(N,θ)|⟩_θ is MAXIMISED when N·δφ/2 = 0 (mod π),
i.e., at the correct CZ operating point with even N it stays flat and high.

Fitting the sinusoidal phase cut D(N*,θ) at peak N* extracts the accumulated
phase φ_acc per CZ gate via a first-Fourier-component projection:

    cos_proj = ⟨D·cos(θ)⟩_θ  →  A/2 · cos(φ_acc)
    sin_proj = ⟨D·sin(θ)⟩_θ  →  A/2 · sin(φ_acc)
    φ_acc (total, N* gates) = atan2(sin_proj, cos_proj)

Phase compensation written to the CZ macro (per gate, in turns):

    phase_shift_target  = −φ_acc_T / (N* · 2π)
    phase_shift_control = −φ_acc_C / (N* · 2π)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class PhaseErrorAmplificationResult:
    """Per-pair result of the 2-D phase error-amplification analysis."""

    # ── Target qubit sweep ──────────────────────────────────────────────
    peak_num_gates: int = 0
    max_mean_abs_diff: float = float("nan")
    mean_contrast: float = float("nan")
    phase_compensation_target: float = float("nan")
    """Per-gate phase correction for the target qubit (turns, for frame_rotation_2pi)."""
    success_target: bool = False

    # ── Control qubit sweep ─────────────────────────────────────────────
    peak_num_gates_ctrl: int = 0
    max_mean_abs_diff_ctrl: float = float("nan")
    mean_contrast_ctrl: float = float("nan")
    phase_compensation_control: float = float("nan")
    """Per-gate phase correction for the control qubit (turns, for frame_rotation_2pi)."""
    success_control: bool = False

    # ── Overall ─────────────────────────────────────────────────────────
    success: bool = False
    """True when both target and control sweeps succeeded."""


def _process_one_sweep(
    data: xr.DataArray,
    num_gates: np.ndarray,
    phases_rad: np.ndarray,
    *,
    state_dim: str = "control_state",
) -> dict[str, Any]:
    """Compute error-amplification metrics and phase compensation for one Ramsey sweep.

    The phase compensation is found by collapsing the N (num_cphase_gates) axis
    for each phase value θ:

        std_N(D(N, θ))  =  std of D over all N at fixed θ
                         ∝  |cos(θ − φ_per_gate)|

    The argmax over θ gives the per-gate phase offset φ_per_gate directly
    (no division by N required, since the phase offset in the Ramsey model
    is the per-gate single-qubit rotation, not the accumulated N-gate total).

    Args:
        data: DataArray with dims (``state_dim``, num_cphase_gates, analysis_phase).
        num_gates: 1-D array of CZ repetition counts.
        phases_rad: 1-D array of analysis phases in radians.
        state_dim: coordinate name that carries the conditional-state axis.

    Returns:
        Dict with keys: ``mean_abs_diff``, ``diff_2d``, ``std_over_n``,
        ``mean_over_n``, ``peak_idx``, ``peak_num_gates``, ``max_mean_abs_diff``,
        ``mean_contrast``, ``phase_compensation``, ``success``.
    """
    s0 = data.sel({state_dim: 0}).values.astype(np.float64)  # (N_gates, N_phases)
    s1 = data.sel({state_dim: 1}).values.astype(np.float64)
    diff = s1 - s0  # (N_gates, N_phases)

    # ── Diagnostic: mean |D| per N (used for success check and plots) ───────
    mean_abs_diff = np.mean(np.abs(diff), axis=-1)  # (N_gates,)
    peak_idx = int(np.argmax(mean_abs_diff))
    peak_gates = int(num_gates[peak_idx])
    max_mad = float(mean_abs_diff[peak_idx])
    mean_contrast = float(np.mean(mean_abs_diff))

    success = bool(
        len(num_gates) >= 3
        and np.isfinite(max_mad)
        and max_mad > 0.05
    )

    # ── Phase compensation: collapse N axis, find minimum-std θ ─────────────
    # std_N(D(N, θ)) is minimised at the θ where D is most consistently zero
    # across all N gate counts — i.e., where the analysis axis is orthogonal
    # to the accumulated single-qubit phase.  The compensation is applied at
    # this θ value directly (per-gate, no N division).
    # mean_N(D(N, θ)) gives the non-alternating (common-phase) component.
    std_over_n = np.std(diff, axis=0)    # (N_phases,)
    mean_over_n = np.mean(diff, axis=0)  # (N_phases,)

    phase_compensation = float("nan")
    if success:
        min_phase_idx = int(np.argmin(std_over_n))
        min_phase_rad = float(phases_rad[min_phase_idx])
        # Per-gate compensation at the minimum-std phase.
        phase_compensation = -min_phase_rad / (2.0 * np.pi)

    return {
        "mean_abs_diff": mean_abs_diff,
        "diff_2d": diff,
        "std_over_n": std_over_n,
        "mean_over_n": mean_over_n,
        "peak_idx": peak_idx,
        "peak_num_gates": peak_gates,
        "max_mean_abs_diff": max_mad,
        "mean_contrast": mean_contrast,
        "phase_compensation": phase_compensation,
        "success": success,
    }


def analyse_phase_error_amplification(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Compute error-amplification metrics for both target and control sweeps.

    Looks for two sets of variables in ``ds_raw``:
      - ``{analysis_signal}_{name}``       — target qubit phase sweep
      - ``{analysis_signal}_ctrl_{name}``  — control qubit phase sweep (optional)

    Args:
        ds_raw: Dataset with dims (control_state, num_cphase_gates, analysis_phase).
        qubit_pairs: List of qubit-pair objects with a ``.name`` attribute.
        analysis_signal: Variable prefix to read from ``ds_raw``.

    Returns:
        ``(ds_fit, fit_results)`` where ``ds_fit`` holds per-N mean |D| arrays
        and the full 2-D diff arrays for both sweeps, and ``fit_results`` maps
        pair name → result dict.
    """
    num_gates = ds_raw.coords["num_cphase_gates"].values.astype(np.int64)
    phases = ds_raw.coords["analysis_phase"].values.astype(np.float64)

    fit_results: dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        target_var = f"{analysis_signal}_{qp.name}"
        ctrl_var = f"{analysis_signal}_ctrl_{qp.name}"

        result = PhaseErrorAmplificationResult()

        # ── Target qubit sweep ──────────────────────────────────────────
        if target_var in ds_raw.data_vars and "analysis_phase" in ds_raw[target_var].dims:
            r = _process_one_sweep(ds_raw[target_var], num_gates, phases)
            result.peak_num_gates = r["peak_num_gates"]
            result.max_mean_abs_diff = r["max_mean_abs_diff"]
            result.mean_contrast = r["mean_contrast"]
            result.phase_compensation_target = r["phase_compensation"]
            result.success_target = r["success"]

            fit_vars[f"mean_abs_diff_{qp.name}"] = xr.DataArray(
                r["mean_abs_diff"],
                dims=["num_cphase_gates"],
                coords={"num_cphase_gates": num_gates},
                attrs={"long_name": "target ⟨|D(N,θ)|⟩_θ", "units": ""},
            )
            fit_vars[f"diff_2d_{qp.name}"] = xr.DataArray(
                r["diff_2d"],
                dims=["num_cphase_gates", "analysis_phase"],
                coords={"num_cphase_gates": num_gates, "analysis_phase": phases},
                attrs={"long_name": "target D = S(ctrl|1⟩)−S(ctrl|0⟩)", "units": ""},
            )
            fit_vars[f"std_over_n_{qp.name}"] = xr.DataArray(
                r["std_over_n"],
                dims=["analysis_phase"],
                coords={"analysis_phase": phases},
                attrs={"long_name": "target std_N(D) vs θ", "units": ""},
            )
            fit_vars[f"mean_over_n_{qp.name}"] = xr.DataArray(
                r["mean_over_n"],
                dims=["analysis_phase"],
                coords={"analysis_phase": phases},
                attrs={"long_name": "target mean_N(D) vs θ", "units": ""},
            )
        else:
            logger.warning("No target sweep data for pair %s; skipping.", qp.name)

        # ── Control qubit sweep ─────────────────────────────────────────
        if ctrl_var in ds_raw.data_vars and "analysis_phase" in ds_raw[ctrl_var].dims:
            r = _process_one_sweep(
                ds_raw[ctrl_var], num_gates, phases, state_dim="control_state"
            )
            result.peak_num_gates_ctrl = r["peak_num_gates"]
            result.max_mean_abs_diff_ctrl = r["max_mean_abs_diff"]
            result.mean_contrast_ctrl = r["mean_contrast"]
            result.phase_compensation_control = r["phase_compensation"]
            result.success_control = r["success"]

            fit_vars[f"mean_abs_diff_ctrl_{qp.name}"] = xr.DataArray(
                r["mean_abs_diff"],
                dims=["num_cphase_gates"],
                coords={"num_cphase_gates": num_gates},
                attrs={"long_name": "control ⟨|D(N,θ)|⟩_θ", "units": ""},
            )
            fit_vars[f"diff_2d_ctrl_{qp.name}"] = xr.DataArray(
                r["diff_2d"],
                dims=["num_cphase_gates", "analysis_phase"],
                coords={"num_cphase_gates": num_gates, "analysis_phase": phases},
                attrs={"long_name": "control D = S(tgt|1⟩)−S(tgt|0⟩)", "units": ""},
            )
            fit_vars[f"std_over_n_ctrl_{qp.name}"] = xr.DataArray(
                r["std_over_n"],
                dims=["analysis_phase"],
                coords={"analysis_phase": phases},
                attrs={"long_name": "control std_N(D) vs θ", "units": ""},
            )
            fit_vars[f"mean_over_n_ctrl_{qp.name}"] = xr.DataArray(
                r["mean_over_n"],
                dims=["analysis_phase"],
                coords={"analysis_phase": phases},
                attrs={"long_name": "control mean_N(D) vs θ", "units": ""},
            )
        else:
            logger.debug("No control sweep data for pair %s.", qp.name)

        result.success = result.success_target and result.success_control
        fit_results[qp.name] = asdict(result)

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"num_cphase_gates": num_gates, "analysis_phase": phases},
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log phase error-amplification results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "PARTIAL/FAILED"
        _log(
            f"  {name}: [{status}] "
            f"target — N*={r['peak_num_gates']}, "
            f"⟨|D|⟩={r['max_mean_abs_diff']:.4f}, "
            f"φ_comp={r['phase_compensation_target']:.5f} turns; "
            f"control — N*={r['peak_num_gates_ctrl']}, "
            f"⟨|D|⟩={r['max_mean_abs_diff_ctrl']:.4f}, "
            f"φ_comp={r['phase_compensation_control']:.5f} turns"
        )