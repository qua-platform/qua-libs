"""Per-qubit-pair RB fitting mechanics.

Survival statistics, exponential curve fit, fit-quality validation, and the
Standard / Interleaved pair fitters. Alpha -> fidelity conversion is delegated
to ``fidelity.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit

from calibration_utils.two_qubit_rb import fidelity

_RESIDUAL_SIGMA_LIMIT = 4.0
_FLOAT_TOLERANCE = 1e-9
# Empirical tolerances for soft (log-only) survival-curve sanity checks — not
# derived from a model; chosen to flag clearly-off data without false-positiving
# on ordinary shot noise.
_MONOTONIC_DECAY_MIN_TOLERANCE = 0.03
_SHALLOW_SURVIVAL_THRESHOLD = 0.35


def rb_decay_curve(x, A, alpha, B):
    """Exponential decay model for RB survival probability."""
    return A * alpha**x + B


def _survival_probability(ds_qp: xr.Dataset) -> xr.DataArray:
    """P(|00>) vs circuit depth, averaged over sequence and shots."""
    survival = (ds_qp.state == 0).mean(dim=["sequence", "shots"])
    if "qubit_pair" in survival.dims:
        survival = survival.squeeze("qubit_pair", drop=True)
    return survival


def _survival_per_sequence(ds_qp: xr.Dataset) -> xr.DataArray:
    """P(|00>) per random sequence at each depth, averaged over shots."""
    survival = (ds_qp.state == 0).mean(dim="shots")
    if "qubit_pair" in survival.dims:
        survival = survival.squeeze("qubit_pair", drop=True)
    return survival


def _survival_stderr(ds_qp: xr.Dataset) -> xr.DataArray:
    """Standard error of the mean survival probability at each circuit depth.

    Shot-averaged survival per random sequence, then SEM across sequences only
    (shots are repeats of the same circuit, not independent RB samples).
    """
    per_sequence = (ds_qp.state == 0).mean(dim="shots")
    stderr = per_sequence.std(dim="sequence") / np.sqrt(ds_qp.sizes["sequence"])
    if "qubit_pair" in stderr.dims:
        stderr = stderr.squeeze("qubit_pair", drop=True)
    return stderr


def _check_survival_soft_warnings(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
) -> list[str]:
    """Soft checks on raw survival data (warnings only; do not fail the fit)."""
    warnings: list[str] = []
    depths = np.asarray(circuit_depths, dtype=float)
    surv = np.asarray(survival, dtype=float)
    err = np.asarray(stderr, dtype=float)

    finite = np.isfinite(surv)
    if not finite.any():
        return warnings

    shallow_idx = int(np.argmin(depths))
    deep_idx = int(np.argmax(depths))
    shallow_depth = float(depths[shallow_idx])
    deep_depth = float(depths[deep_idx])
    shallow_survival = float(surv[shallow_idx])
    deep_survival = float(surv[deep_idx])

    if deep_depth > shallow_depth and np.isfinite(deep_survival) and np.isfinite(shallow_survival):
        tolerance = max(
            float(err[shallow_idx]) if np.isfinite(err[shallow_idx]) else 0.0,
            float(err[deep_idx]) if np.isfinite(err[deep_idx]) else 0.0,
            _MONOTONIC_DECAY_MIN_TOLERANCE,
        )
        if deep_survival > shallow_survival + tolerance:
            warnings.append(
                f"P(|00>) at max depth {deep_depth:g} ({deep_survival:.3f}) exceeds "
                f"shallow depth {shallow_depth:g} ({shallow_survival:.3f}) by more than "
                f"{tolerance:.3f} — decay is not monotonic; check SPAM/readout or shot count."
            )

    if np.isfinite(shallow_survival) and shallow_survival < _SHALLOW_SURVIVAL_THRESHOLD:
        warnings.append(
            f"P(|00>) at depth {shallow_depth:g} is {shallow_survival:.3f} (well below 0.5) — "
            "suggests readout, reset, or SPAM issues; fitted fidelity may be unreliable."
        )

    return warnings


def _fit_survival(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Fit ``A * alpha**m + B`` and return ``(A, alpha, B, alpha_stderr, A_stderr, B_stderr)``."""
    sigma = None
    if stderr is not None:
        err = np.asarray(stderr, dtype=float)
        if err.shape == np.asarray(survival).shape and np.all(np.isfinite(err)) and np.all(err > 0):
            sigma = err

    try:
        popt, pcov = curve_fit(
            rb_decay_curve,
            circuit_depths,
            survival,
            p0=[0.75, 0.9, 0.25],
            maxfev=10000,
            sigma=sigma,
            absolute_sigma=sigma is not None,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("RB exponential fit failed: %s", exc)
        return None

    alpha_stderr = amplitude_stderr = offset_stderr = np.nan
    if pcov is not None and pcov.shape == (3, 3):
        if np.isfinite(pcov[1, 1]) and pcov[1, 1] >= 0:
            alpha_stderr = float(np.sqrt(pcov[1, 1]))
        if np.isfinite(pcov[0, 0]) and pcov[0, 0] >= 0:
            amplitude_stderr = float(np.sqrt(pcov[0, 0]))
        if np.isfinite(pcov[2, 2]) and pcov[2, 2] >= 0:
            offset_stderr = float(np.sqrt(pcov[2, 2]))

    return (
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        alpha_stderr,
        amplitude_stderr,
        offset_stderr,
    )


def _validate_fit_parameters(fit_amplitude: float, alpha: float, fit_offset: float) -> list[str]:
    """Hard-fail checks on exponential-fit coefficients."""
    issues: list[str] = []
    if not all(np.isfinite(v) for v in (fit_amplitude, alpha, fit_offset)):
        issues.append("Non-finite fit parameters (A, alpha, or B).")
        return issues

    if alpha <= _FLOAT_TOLERANCE or alpha > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"RB decay alpha={alpha:.6f} outside physical range (0, 1].")

    if fit_amplitude < -_FLOAT_TOLERANCE:
        issues.append(f"Fit amplitude A={fit_amplitude:.6f} must be >= 0.")

    if fit_offset < -_FLOAT_TOLERANCE:
        issues.append(f"Fit offset B={fit_offset:.6f} must be >= 0.")

    if fit_amplitude + fit_offset > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"A + B = {fit_amplitude + fit_offset:.6f} exceeds 1 (invalid survival-probability model).")
    return issues


def _validate_residuals(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
) -> list[str]:
    """Hard-fail when the fitted curve deviates too far from the data."""
    fitted_values = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    safe_stderr = np.where(stderr > 0, stderr, np.nan)
    normalized_residuals = np.abs(fitted_values - survival) / safe_stderr
    if not np.isfinite(normalized_residuals).any():
        return ["Cannot assess fit residuals (zero or missing standard errors)."]

    max_deviation = float(np.nanmax(normalized_residuals))
    if max_deviation > _RESIDUAL_SIGMA_LIMIT:
        return [
            f"Fitted curve deviates up to {max_deviation:.2f} sigma from experimental data "
            f"(limit {_RESIDUAL_SIGMA_LIMIT:.0f} sigma)."
        ]
    return []


def _validate_rb_fit(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
    *,
    interleaved: bool,
    fit_fidelity: float,
    standard_rb_alpha: float | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Run all hard-fail RB fit validations."""
    issues: list[str] = []
    issues.extend(_validate_fit_parameters(fit_amplitude, alpha, fit_offset))
    issues.extend(fidelity.validate_fidelity_bounds(fit_fidelity, interleaved=interleaved))
    if interleaved and standard_rb_alpha is not None:
        issues.extend(fidelity.validate_interleaved_alpha(alpha, standard_rb_alpha))
    if not issues:
        issues.extend(_validate_residuals(circuit_depths, survival, stderr, fit_amplitude, alpha, fit_offset))
    return len(issues) == 0, tuple(issues)


def _fit_survival_for_pair(da: xr.Dataset) -> dict:
    """Compute survival stats + exponential fit for one qubit pair."""
    survival = _survival_probability(da)
    survival_per_sequence = _survival_per_sequence(da)
    stderr = _survival_stderr(da)
    circuit_depths = survival.circuit_depth.values
    survival_vals = survival.values
    stderr_vals = stderr.values

    fit_warnings = tuple(_check_survival_soft_warnings(circuit_depths, survival_vals, stderr_vals))
    fit_params = _fit_survival(circuit_depths, survival_vals, stderr_vals)

    return {
        "survival": survival,
        "survival_per_sequence": survival_per_sequence,
        "stderr": stderr,
        "circuit_depths": circuit_depths,
        "survival_vals": survival_vals,
        "stderr_vals": stderr_vals,
        "fit_warnings": fit_warnings,
        "fit_params": fit_params,
    }


def _fitted_curve_dataarray(circuit_depths, fit_params) -> xr.DataArray:
    if fit_params is not None:
        fit_amplitude, alpha, fit_offset, _, _, _ = fit_params
        fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    else:
        fitted_curve = np.full_like(circuit_depths, np.nan, dtype=float)
    return xr.DataArray(fitted_curve, dims=["circuit_depth"], coords={"circuit_depth": circuit_depths})


def _try_load_standard_rb_overlay(node: QualibrationNode, qp_name: str) -> dict | None:
    """Load saved 37a ``ds_fit`` for interleaved overlay plots (no refit)."""
    standard_rb_load_id = (
        node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity.get("StandardRB_load_id")
    )
    if standard_rb_load_id is None:
        return None

    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    try:
        base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
        node_dir = get_node_dir_path(int(standard_rb_load_id), base_path)
        std_rb_data = read_node_data(node_dir, int(standard_rb_load_id), base_path)
    except Exception as exc:
        node.log(f"Could not load StandardRB overlay for {qp_name}: failed to read saved node data ({exc}).")
        return None

    if std_rb_data is None:
        return None

    ds_fit = std_rb_data.get("ds_fit")
    if ds_fit is None or qp_name not in ds_fit.qubit_pair.values:
        node.log(
            f"Could not load StandardRB overlay for {qp_name}: "
            f"saved 37a run {standard_rb_load_id} has no ds_fit for this pair."
        )
        return None

    fr = ds_fit.sel(qubit_pair=qp_name)
    if "survival_probability" not in fr or "fitted_curve" not in fr or "fit_alpha" not in fr:
        node.log(
            f"Could not load StandardRB overlay for {qp_name}: "
            f"saved 37a ds_fit is missing survival or fit variables."
        )
        return None

    alpha_stderr = np.nan
    if "alpha_stderr" in fr and np.isfinite(fr.alpha_stderr.values):
        alpha_stderr = float(fr.alpha_stderr.values)

    epc_reference = None
    if "epc" in fr and np.isfinite(fr.epc.values):
        epc_reference = float(fr.epc.values)

    survival_per_sequence = fr.survival_per_sequence if "survival_per_sequence" in fr else None

    return {
        "circuit_depth": np.asarray(fr.circuit_depth.values, dtype=float),
        "survival": np.asarray(fr.survival_probability.values, dtype=float),
        "survival_per_sequence": survival_per_sequence,
        "fitted_curve": np.asarray(fr.fitted_curve.values, dtype=float),
        "alpha": float(fr.fit_alpha.values),
        "alpha_stderr": alpha_stderr,
        "epc_reference": epc_reference,
    }


def _resolve_irb_reference(
    node: QualibrationNode, qp_name: str, overlay: dict | None
) -> tuple[float, float | None, float | None]:
    """Resolve reference Standard RB alpha and EPC for interleaved analysis.

    Prefer saved 37a ``ds_fit`` overlay; fall back to QUAM.
    """
    if overlay is not None:
        alpha_stderr = overlay.get("alpha_stderr")
        stderr = (
            float(alpha_stderr) if alpha_stderr is not None and np.isfinite(alpha_stderr) and alpha_stderr > 0 else None
        )
        return float(overlay["alpha"]), stderr, overlay.get("epc_reference")

    fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
    if "StandardRB_alpha" not in fidelity_dict:
        raise KeyError(
            f"Qubit pair {qp_name}: missing StandardRB_alpha in "
            f"macros[{node.parameters.operation!r}].fidelity. "
            "Run 37a_two_qubit_standard_rb first for this operation."
        )
    standard_rb_alpha = float(fidelity_dict["StandardRB_alpha"])
    epc_reference = 1 - float(fidelity_dict["StandardRB"]) if "StandardRB" in fidelity_dict else None
    return standard_rb_alpha, None, epc_reference


def _assign_standard_rb_overlay(
    da: xr.Dataset,
    overlay: dict,
    *,
    node: QualibrationNode | None = None,
    qp_name: str | None = None,
) -> xr.Dataset:
    """Align a Standard RB overlay onto the interleaved dataset circuit depths."""
    circuit_depths = da.circuit_depth.values
    survival_on_depths = np.full(len(circuit_depths), np.nan)
    fitted_on_depths = np.full(len(circuit_depths), np.nan)
    overlay_depths = overlay["circuit_depth"]
    overlay_per_sequence = overlay.get("survival_per_sequence")
    n_sequence = da.sizes.get("sequence")
    per_sequence_on_depths = None
    if overlay_per_sequence is not None and n_sequence is not None:
        overlay_n_sequence = overlay_per_sequence.sizes.get("sequence")
        if (
            overlay_n_sequence == n_sequence
        ):  # if the number of sequences is the same, then we can use the per-sequence data
            per_sequence_on_depths = np.full((len(circuit_depths), n_sequence), np.nan)
        elif node is not None:
            node.log(
                f"Standard RB overlay"
                f"{f' for {qp_name}' if qp_name else ''}: skipping per-sequence points "
                f"(saved 37a num_circuits_per_depth={overlay_n_sequence} != "
                f"current num_circuits_per_depth={n_sequence}). Depth-mean overlay kept."
            )

    for idx, depth in enumerate(circuit_depths):
        match = np.where(overlay_depths == depth)[0]
        if match.size:
            survival_on_depths[idx] = overlay["survival"][match[0]]
            fitted_on_depths[idx] = overlay["fitted_curve"][match[0]]
            if per_sequence_on_depths is not None:
                per_sequence_on_depths[idx, :] = overlay_per_sequence.isel(circuit_depth=match[0]).values

    assign_kwargs = {
        "standard_rb_overlay_survival": ("circuit_depth", survival_on_depths),
        "standard_rb_overlay_fitted": ("circuit_depth", fitted_on_depths),
        "standard_rb_fit_alpha": float(overlay["alpha"]),
        "standard_rb_fit_alpha_stderr": float(overlay.get("alpha_stderr", np.nan)),
    }
    if per_sequence_on_depths is not None:
        assign_kwargs["standard_rb_overlay_survival_per_sequence"] = (
            ["circuit_depth", "sequence"],
            per_sequence_on_depths,
        )

    return da.assign(**assign_kwargs)


def fit_srb_pair(da: xr.Dataset, average_gates_per_clifford: float | None) -> xr.Dataset:
    """Fit Standard RB survival probability vs circuit depth for one qubit pair."""
    stats = _fit_survival_for_pair(da)
    circuit_depths, survival_vals, stderr_vals = stats["circuit_depths"], stats["survival_vals"], stats["stderr_vals"]
    fit_params = stats["fit_params"]

    if fit_params is None:
        fit_amplitude = alpha = fit_offset = alpha_stderr = amplitude_stderr = offset_stderr = np.nan
        srb = fidelity.SRBFidelity(
            fidelity=np.nan, fidelity_stderr=None, epc=np.nan, epg=None, average_gate_fidelity=None
        )
        success, fit_issues = False, ("Exponential fit did not converge.",)
    else:
        fit_amplitude, alpha, fit_offset, alpha_stderr, amplitude_stderr, offset_stderr = fit_params
        srb = fidelity.compute_srb_fidelity(alpha, alpha_stderr, average_gates_per_clifford)
        success, fit_issues = _validate_rb_fit(
            circuit_depths,
            survival_vals,
            stderr_vals,
            fit_amplitude,
            alpha,
            fit_offset,
            interleaved=False,
            fit_fidelity=srb.fidelity,
        )

    fitted_curve_da = _fitted_curve_dataarray(circuit_depths, fit_params)

    assign_kwargs = {
        "survival_probability": stats["survival"],
        "survival_per_sequence": stats["survival_per_sequence"],
        "survival_stderr": stats["stderr"],
        "fitted_curve": fitted_curve_da,
        "fit_amplitude": fit_amplitude,
        "fit_alpha": alpha,
        "fit_offset": fit_offset,
        "alpha_stderr": alpha_stderr,
        "fit_amplitude_stderr": amplitude_stderr,
        "fit_offset_stderr": offset_stderr,
        "fidelity": srb.fidelity,
        "fidelity_stderr": srb.fidelity_stderr if srb.fidelity_stderr is not None else np.nan,
        "epc": srb.epc,
        "epg": srb.epg if srb.epg is not None else np.nan,
        "success": success,
        "fit_issues": "\n".join(fit_issues),
        "fit_warnings": "\n".join(stats["fit_warnings"]),
    }
    if srb.average_gate_fidelity is not None:
        assign_kwargs["average_gate_fidelity"] = srb.average_gate_fidelity
    if average_gates_per_clifford is not None:
        assign_kwargs["average_gates_per_clifford"] = average_gates_per_clifford

    return da.assign(**assign_kwargs)


def fit_irb_pair(da: xr.Dataset, node: QualibrationNode, qp_name: str) -> xr.Dataset:
    """Fit Interleaved RB survival probability vs circuit depth for one qubit pair."""
    overlay = _try_load_standard_rb_overlay(node, qp_name)
    standard_rb_alpha, standard_rb_alpha_stderr, epc_reference = _resolve_irb_reference(node, qp_name, overlay)

    stats = _fit_survival_for_pair(da)
    circuit_depths, survival_vals, stderr_vals = stats["circuit_depths"], stats["survival_vals"], stats["stderr_vals"]
    fit_params = stats["fit_params"]

    if fit_params is None:
        fit_amplitude = alpha = fit_offset = alpha_stderr = amplitude_stderr = offset_stderr = np.nan
        irb = fidelity.IRBFidelity(fidelity=np.nan, fidelity_stderr=None, epc=np.nan, epg=np.nan, epg_stderr=None)
        success, fit_issues = False, ("Exponential fit did not converge.",)
    else:
        fit_amplitude, alpha, fit_offset, alpha_stderr, amplitude_stderr, offset_stderr = fit_params
        irb = fidelity.compute_irb_fidelity(alpha, alpha_stderr, standard_rb_alpha, epc_reference=epc_reference)
        success, fit_issues = _validate_rb_fit(
            circuit_depths,
            survival_vals,
            stderr_vals,
            fit_amplitude,
            alpha,
            fit_offset,
            interleaved=True,
            fit_fidelity=irb.fidelity,
            standard_rb_alpha=standard_rb_alpha,
        )

    fitted_curve_da = _fitted_curve_dataarray(circuit_depths, fit_params)

    da = da.assign(
        survival_probability=stats["survival"],
        survival_per_sequence=stats["survival_per_sequence"],
        survival_stderr=stats["stderr"],
        fitted_curve=fitted_curve_da,
        fit_amplitude=fit_amplitude,
        fit_alpha=alpha,
        fit_offset=fit_offset,
        alpha_stderr=alpha_stderr,
        fit_amplitude_stderr=amplitude_stderr,
        fit_offset_stderr=offset_stderr,
        fidelity=irb.fidelity,
        fidelity_stderr=irb.fidelity_stderr if irb.fidelity_stderr is not None else np.nan,
        epc=irb.epc,
        epg=irb.epg,
        success=success,
        fit_issues="\n".join(fit_issues),
        fit_warnings="\n".join(stats["fit_warnings"]),
        standard_rb_alpha=standard_rb_alpha,
        standard_rb_fit_alpha_stderr=standard_rb_alpha_stderr if standard_rb_alpha_stderr is not None else np.nan,
    )

    if overlay is not None:
        da = _assign_standard_rb_overlay(da, overlay, node=node, qp_name=qp_name)
    else:
        nan_overlay = np.full(len(circuit_depths), np.nan)
        da = da.assign(
            standard_rb_overlay_survival=("circuit_depth", nan_overlay),
            standard_rb_overlay_fitted=("circuit_depth", nan_overlay.copy()),
            standard_rb_fit_alpha=np.nan,
            standard_rb_fit_alpha_stderr=np.nan,
        )
    return da


__all__ = ["rb_decay_curve", "fit_srb_pair", "fit_irb_pair"]
