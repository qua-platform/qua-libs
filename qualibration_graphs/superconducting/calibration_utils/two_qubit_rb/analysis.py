"""Analysis utilities for two-qubit randomized benchmarking experiments.

Flow (called from ``37a`` / ``37b`` ``analyse_data``)
-----------------------------------------------------

1. **``process_raw_dataset``** — normalize ``ds_raw`` layout. Input-stream fetches use
   ``(circuit_depth, shots, sequence)``; this transposes to canonical
   ``(shots, circuit_depth, sequence)`` so one analysis path serves both execution modes.

2. **``fit_raw_data``** — group by ``qubit_pair`` and run ``fit_rb_routine`` per pair,
   then build a ``FitResults`` dict for logging and node outcomes.

Per qubit pair, **``fit_rb_routine``** does:

3. **Survival statistics** — from classified ``state`` data, compute:
   - mean P(|00⟩) vs depth (``survival_probability``),
   - per-random-sequence survival (``survival_per_sequence``),
   - SEM at each depth (``survival_stderr``).

4. **Soft data checks** (``_check_survival_soft_warnings``) — log-only warnings:
   - non-monotonic shallow vs max depth (possible SPAM/readout / low shots),
   - very low P(|00⟩) at the shallowest depth (readout / reset / SPAM).

5. **Exponential fit** — ``curve_fit`` to ``A * alpha**m + B`` vs circuit depth.

6. **Fidelity extraction**
   - **Standard RB (37a):** 2Q Clifford fidelity from ``alpha``; EPC; per-gate EPG as
     EPC divided by measured average native gates per Clifford from transpilation.
   - **Interleaved RB (37b):** CZ gate fidelity from interleaved ``alpha`` and reference
     ``StandardRB_alpha`` in QUAM; EPG = 1 − CZ fidelity; EPC from reference Standard RB.

7. **Hard validation** (``_validate_rb_fit``) — sets ``success=False`` (skips QUAM update) when:
   - fit fails to converge,
   - ``alpha`` or A/B coefficients are unphysical,
   - fidelity ∉ [0, 1],
   - interleaved ``alpha`` > reference ``StandardRB_alpha`` (CZ fidelity > 100%),
   - fitted curve deviates > 4σ from data.
   Reasons are stored in ``fit_issues``.

8. **Interleaved overlay (37b only)** — optionally reload the reference Standard RB run
   (``StandardRB_load_id``) and attach its survival / fit curves for plotting.

Outputs: augmented ``ds_fit`` (survival, fit curve, fidelity, EPC/EPG, ``success``,
``fit_issues``, ``fit_warnings``) and ``FitResults`` per pair. Nodes set ``outcomes``
from ``success``; ``log_fitted_results`` prints metrics, hard failures, and warnings.
Plotting (``plotting.py``) draws the exponential fit only when ``success`` is True.
"""

# pylint: disable=use-implicit-booleaness-not-comparison-to-zero

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit

from calibration_utils.two_qubit_rb.coherence_limit import try_coherence_limit_epg

_RESIDUAL_SIGMA_LIMIT = 4.0
_FLOAT_TOLERANCE = 1e-9
_MONOTONIC_DECAY_MIN_TOLERANCE = 0.03
_SHALLOW_SURVIVAL_THRESHOLD = 0.35


@dataclass
class FitResults:
    """Stores the relevant RB fit parameters for a single qubit pair."""

    alpha: float
    fidelity: float
    success: bool
    fit_amplitude: float
    fit_offset: float
    epc: float | None = None
    epg: float | None = None
    average_gate_fidelity: float | None = None
    average_gates_per_clifford: float | None = None
    standard_rb_alpha: float | None = None
    epc_1q_budget: float | None = None
    epc_cz_residual: float | None = None
    epg_cz_implied: float | None = None
    f_1q_control: float | None = None
    f_1q_target: float | None = None
    avg_1q_per_clifford: float | None = None
    avg_cz_per_clifford: float | None = None
    coherence_limit_epg: float | None = None
    fit_issues: tuple[str, ...] = field(default_factory=tuple)
    fit_warnings: tuple[str, ...] = field(default_factory=tuple)


def format_error_rate(rate: float | None) -> str:
    """Format EPC/EPG for logs and plot annotations."""
    if rate is None or not np.isfinite(rate):
        return "n/a"
    if rate < 0.01:
        return f"{rate * 1e3:.2f} × 10⁻³"
    return f"{100 * rate:.2f}%"


def log_fitted_results(
    fit_results: Dict[str, FitResults],
    log_callable=None,
    *,
    interleaved: bool = False,
):
    """Log fitted RB results for all qubit pairs."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        lines = [f"Results for qubit pair {qp_name}: {'SUCCESS!' if fit_result.success else 'FAIL!'}"]

        if fit_result.success:
            lines.append(
                f"\tDecay model: P(m) = A * alpha^m + B "
                f"(A = {fit_result.fit_amplitude:.6f}, alpha = {fit_result.alpha:.6f}, "
                f"B = {fit_result.fit_offset:.6f})"
            )
            if interleaved:
                clifford_fid_ref = 1 - fit_result.epc if fit_result.epc is not None else np.nan
                lines.extend(
                    [
                        f"\t2Q Clifford Fidelity (Standard RB reference) = {100 * clifford_fid_ref:.2f}%",
                        f"\tError Per Clifford (EPC) = 1 - 2Q Clifford Fidelity = "
                        f"{format_error_rate(fit_result.epc)}",
                        f"\tCZ gate fidelity = 1 - (d-1)/d * (1 - alpha_IRB/alpha_SRB), d = 4 = "
                        f"{100 * fit_result.fidelity:.2f}%, "
                        f"alpha_IRB = {fit_result.alpha:.6f}, alpha_SRB = {fit_result.standard_rb_alpha:.6f}",
                        f"\tError Per Gate (EPG) = 1 - CZ gate fidelity = {format_error_rate(fit_result.epg)}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"\t2Q Clifford Fidelity = {100 * fit_result.fidelity:.2f}%",
                        f"\tError Per Clifford (EPC): 1 - 2Q Clifford Fidelity = "
                        f"{format_error_rate(fit_result.epc)}",
                        f"\tError Per Gate (EPG) = EPC / N_gates_per_Clifford = "
                        f"{format_error_rate(fit_result.epc)} / {fit_result.average_gates_per_clifford:.2f} = "
                        f"{format_error_rate(fit_result.epg)}",
                        f"\tAvg. Gate Fidelity (1-EPG) = {100 * fit_result.average_gate_fidelity:.2f}%",
                    ]
                )

            if not interleaved and fit_result.epc_1q_budget is not None:
                epc = fit_result.epc
                epc_1q = fit_result.epc_1q_budget
                epc_cz = fit_result.epc_cz_residual
                lines.extend(
                    [
                        "",
                        "\tError contribution analysis:",
                        f"\t1Q RB budget: EPC_1Q = 1 - F_1Q,c^(N_1Q/2) * F_1Q,t^(N_1Q/2) = "
                        f"{format_error_rate(epc_1q)} "
                        f"(N_1Q = {fit_result.avg_1q_per_clifford:.2f}, "
                        f"F_1Q,c = {100 * fit_result.f_1q_control:.2f}%, "
                        f"F_1Q,t = {100 * fit_result.f_1q_target:.2f}%)",
                        f"\tCZ residual: EPC_CZ = EPC - EPC_1Q = {format_error_rate(epc_cz)} "
                        f"(N_CZ = {fit_result.avg_cz_per_clifford:.2f})",
                        f"\tEPC contribution: 1Q = {100 * epc_1q / epc:.1f}%, "
                        f"CZ = {100 * epc_cz / epc:.1f}%",
                        f"\tImplied CZ EPG = 1 - (F_Clifford / F_1Q_budget)^(1/N_CZ) = "
                        f"{format_error_rate(fit_result.epg_cz_implied)}",
                        "\tNote: Interleaved RB (37b) measures CZ EPG directly; treat this implied value as a rough estimate only.",
                    ]
                )

            if fit_result.coherence_limit_epg is not None:
                lines.append(
                    f"\tCoherence-limited EPG (T1/T2 floor) = "
                    f"{format_error_rate(fit_result.coherence_limit_epg)}"
                )
                if interleaved and fit_result.epg is not None and fit_result.coherence_limit_epg > 0:
                    ratio = fit_result.epg / fit_result.coherence_limit_epg
                    verdict = (
                        "Not coherence-limited (EPG exceeds T1/T2 floor)"
                        if fit_result.epg > fit_result.coherence_limit_epg
                        else "Coherence-limited (EPG at or below T1/T2 floor)"
                    )
                    lines.append(
                        f"\t{verdict} "
                        f"({ratio:.1f}× floor: {format_error_rate(fit_result.epg)} vs "
                        f"{format_error_rate(fit_result.coherence_limit_epg)})"
                    )

        for issue in fit_result.fit_issues:
            lines.append(f"\t- {issue}")
        for warning in fit_result.fit_warnings:
            lines.append(f"\tWarning: {warning}")

        log_callable("\n".join(lines))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode | None = None) -> xr.Dataset:
    """Normalize raw RB dataset layout for downstream analysis.

    The input-stream QUA path uses a chunk-outer, shot-inner loop so the OPX can
    replay each pushed sub-chunk for all shots without extra host pushes. That
    measurement order is ``(circuit_depth, shots, sequence)``, and
    ``build_sweep_axes`` matches it at fetch time so ``XarrayDataFetcher`` can
    reshape the raw buffers correctly.

    All RB analysis (survival averaging, fitting, plotting) is written against
    the canonical layout used by the non-input-stream path:
    ``(shots, circuit_depth, sequence)``. This function transposes input-stream
    datasets to that order after fetch so both execution modes share the same
    analysis code without duplicating fit logic or slowing execution by pushing
    one chunk per shot.
    """
    if node is not None and node.parameters.use_input_stream:
        for name in ds.data_vars:
            dims = list(ds[name].dims)
            if {"circuit_depth", "shots", "sequence"}.issubset(dims):
                other_dims = [d for d in dims if d not in ("circuit_depth", "shots", "sequence")]
                ds[name] = ds[name].transpose(*other_dims, "shots", "circuit_depth", "sequence")
    return ds


def rb_decay_curve(x, A, alpha, B):
    """Exponential decay model for RB survival probability."""
    return A * alpha**x + B


def clifford_fidelity_from_alpha(alpha: float, n_qubits: int = 2) -> float:
    """Average Clifford fidelity from the fitted RB decay constant."""
    d = 2**n_qubits
    r = 1 - alpha - (1 - alpha) / d
    return 1 - r


def interleaved_gate_fidelity_from_alpha(alpha: float, standard_rb_alpha: float, n_qubits: int = 2) -> float:
    """Interleaved gate fidelity using https://arxiv.org/pdf/1210.7011."""
    return 1 - ((2**n_qubits - 1) * (1 - alpha / standard_rb_alpha) / 2**n_qubits)


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
    """Standard error of the mean survival probability at each circuit depth."""
    n_samples = ds_qp.sizes["sequence"] * ds_qp.sizes["shots"]
    stderr = (ds_qp.state == 0).stack(combined=("shots", "sequence")).std(dim="combined") / np.sqrt(n_samples)
    if "qubit_pair" in stderr.dims:
        stderr = stderr.squeeze("qubit_pair", drop=True)
    return stderr


def _fit_survival(circuit_depths: np.ndarray, survival: np.ndarray) -> tuple[float, float, float] | None:
    try:
        popt, _ = curve_fit(
            rb_decay_curve,
            circuit_depths,
            survival,
            p0=[0.75, 0.9, 0.25],
            maxfev=10000,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("RB exponential fit failed: %s", exc)
        return None
    return float(popt[0]), float(popt[1]), float(popt[2])


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
        issues.append(f"A + B = {fit_amplitude + fit_offset:.6f} exceeds 1 " "(invalid survival-probability model).")
    return issues


def _validate_fidelity_bounds(fidelity: float, *, interleaved: bool) -> list[str]:
    """Hard-fail checks on extracted gate / Clifford fidelity."""
    label = "CZ gate fidelity" if interleaved else "2Q Clifford fidelity"
    issues: list[str] = []
    if not np.isfinite(fidelity):
        issues.append(f"{label} is non-finite.")
    elif fidelity < -_FLOAT_TOLERANCE or fidelity > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"{label}={100 * fidelity:.4f}% outside physical range [0, 100%].")
    return issues


def _validate_interleaved_alpha(alpha: float, standard_rb_alpha: float) -> list[str]:
    """Hard-fail checks specific to interleaved RB reference comparison."""
    issues: list[str] = []
    if not np.isfinite(standard_rb_alpha):
        issues.append("Reference StandardRB_alpha is non-finite.")
        return issues

    if standard_rb_alpha <= _FLOAT_TOLERANCE or standard_rb_alpha > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"Reference StandardRB_alpha={standard_rb_alpha:.6f} outside physical range (0, 1].")

    if alpha > standard_rb_alpha + _FLOAT_TOLERANCE:
        issues.append(
            f"Interleaved alpha={alpha:.6f} exceeds reference StandardRB_alpha="
            f"{standard_rb_alpha:.6f} (implies CZ fidelity > 100%)."
        )
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


def _validate_rb_fit(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
    *,
    interleaved: bool,
    fidelity: float,
    standard_rb_alpha: float | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Run all hard-fail RB fit validations and return success plus issue messages."""
    issues: list[str] = []
    issues.extend(_validate_fit_parameters(fit_amplitude, alpha, fit_offset))
    issues.extend(_validate_fidelity_bounds(fidelity, interleaved=interleaved))
    if interleaved and standard_rb_alpha is not None:
        issues.extend(_validate_interleaved_alpha(alpha, standard_rb_alpha))
    if not issues:
        issues.extend(
            _validate_residuals(
                circuit_depths,
                survival,
                stderr,
                fit_amplitude,
                alpha,
                fit_offset,
            )
        )
    return len(issues) == 0, tuple(issues)


def _try_load_standard_rb_overlay(node: QualibrationNode, qp_name: str) -> dict | None:
    """Load and fit a reference Standard RB dataset for interleaved overlay plots."""
    standard_rb_load_id = (
        node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity.get("StandardRB_load_id")
    )
    if standard_rb_load_id is None:
        return None

    try:
        from qualibrate.core.utils.node.content import read_node_data
        from qualibrate.core.utils.node.path_solver import get_node_dir_path
        from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

        base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
        node_dir = get_node_dir_path(int(standard_rb_load_id), base_path)
        std_rb_data = read_node_data(node_dir, int(standard_rb_load_id), base_path)
        std_rb_ds = process_raw_dataset(std_rb_data["ds_raw"].sel(qubit_pair=qp_name), node)

        survival = _survival_probability(std_rb_ds)
        circuit_depths = survival.circuit_depth.values
        survival_vals = survival.values
        fit_amplitude, alpha, fit_offset = _fit_survival(circuit_depths, survival_vals)
        fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)

        return {
            "circuit_depth": circuit_depths,
            "survival": survival_vals,
            "survival_per_sequence": _survival_per_sequence(std_rb_ds),
            "fitted_curve": fitted_curve,
            "alpha": alpha,
        }
    except Exception as exc:
        node.log(f"Could not load StandardRB overlay for {qp_name}: {exc}")
        return None


def _assign_standard_rb_overlay(da: xr.Dataset, overlay: dict) -> xr.Dataset:
    """Align a Standard RB overlay onto the interleaved dataset circuit depths."""
    circuit_depths = da.circuit_depth.values
    survival_on_depths = np.full(len(circuit_depths), np.nan)
    fitted_on_depths = np.full(len(circuit_depths), np.nan)
    overlay_depths = overlay["circuit_depth"]
    overlay_per_sequence = overlay.get("survival_per_sequence")
    per_sequence_on_depths = None
    if overlay_per_sequence is not None:
        per_sequence_on_depths = np.full(
            (len(circuit_depths), overlay_per_sequence.sizes["sequence"]),
            np.nan,
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
    }
    if per_sequence_on_depths is not None:
        assign_kwargs["standard_rb_overlay_survival_per_sequence"] = (
            ["circuit_depth", "sequence"],
            per_sequence_on_depths,
        )

    return da.assign(**assign_kwargs)


def fit_rb_routine(da: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Fit RB survival probability vs circuit depth for one qubit pair."""
    interleaved = "interleaved" in node.name.lower()
    average_gates_per_clifford = node.namespace.get("average_gates_per_clifford")
    qp_name = str(np.asarray(da.qubit_pair.values).item())

    survival = _survival_probability(da)
    survival_per_sequence = _survival_per_sequence(da)
    stderr = _survival_stderr(da)
    circuit_depths = survival.circuit_depth.values
    survival_vals = survival.values
    stderr_vals = stderr.values
    fit_warnings = tuple(_check_survival_soft_warnings(circuit_depths, survival_vals, stderr_vals))

    fit_params = _fit_survival(circuit_depths, survival_vals)
    if fit_params is None:
        fit_amplitude = np.nan
        alpha = np.nan
        fit_offset = np.nan
        fidelity = np.nan
        epc = np.nan
        epg = np.nan
        average_gate_fidelity = None
        standard_rb_alpha = None
        if interleaved:
            fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
            if "StandardRB_alpha" not in fidelity_dict:
                raise KeyError(
                    f"Qubit pair {qp_name}: missing StandardRB_alpha in "
                    f"macros[{node.parameters.operation!r}].fidelity. "
                    "Run 37a_two_qubit_standard_rb first for this operation."
                )
            standard_rb_alpha = float(fidelity_dict["StandardRB_alpha"])
        success, fit_issues = False, ("Exponential fit did not converge.",)
    else:
        fit_amplitude, alpha, fit_offset = fit_params
        standard_rb_alpha = None

        if interleaved:
            fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
            if "StandardRB_alpha" not in fidelity_dict:
                raise KeyError(
                    f"Qubit pair {qp_name}: missing StandardRB_alpha in "
                    f"macros[{node.parameters.operation!r}].fidelity. "
                    "Run 37a_two_qubit_standard_rb first for this operation."
                )
            standard_rb_alpha = float(fidelity_dict["StandardRB_alpha"])
            fidelity = interleaved_gate_fidelity_from_alpha(alpha, standard_rb_alpha)
            epg = 1 - fidelity
            if "StandardRB" in fidelity_dict:
                epc = 1 - float(fidelity_dict["StandardRB"])
            else:
                epc = 1 - clifford_fidelity_from_alpha(standard_rb_alpha)
            average_gate_fidelity = None
        else:
            fidelity = clifford_fidelity_from_alpha(alpha)
            epc = 1 - fidelity
            if average_gates_per_clifford is not None and average_gates_per_clifford > 0:
                epg = (1 - fidelity) / average_gates_per_clifford
                average_gate_fidelity = 1 - epg
            else:
                epg = np.nan
                average_gate_fidelity = None

        success, fit_issues = _validate_rb_fit(
            circuit_depths,
            survival_vals,
            stderr_vals,
            fit_amplitude,
            alpha,
            fit_offset,
            interleaved=interleaved,
            fidelity=fidelity,
            standard_rb_alpha=standard_rb_alpha,
        )

    if fit_params is not None:
        fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    else:
        fitted_curve = np.full_like(circuit_depths, np.nan, dtype=float)
    fitted_curve_da = xr.DataArray(
        fitted_curve,
        dims=["circuit_depth"],
        coords={"circuit_depth": circuit_depths},
    )

    assign_kwargs = {
        "survival_probability": survival,
        "survival_per_sequence": survival_per_sequence,
        "survival_stderr": stderr,
        "fitted_curve": fitted_curve_da,
        "fit_amplitude": fit_amplitude,
        "fit_alpha": alpha,
        "fit_offset": fit_offset,
        "fidelity": fidelity,
        "epc": epc,
        "epg": epg,
        "success": success,
        "fit_issues": "\n".join(fit_issues),
        "fit_warnings": "\n".join(fit_warnings),
    }
    if average_gate_fidelity is not None:
        assign_kwargs["average_gate_fidelity"] = average_gate_fidelity
    if not interleaved and average_gates_per_clifford is not None:
        assign_kwargs["average_gates_per_clifford"] = average_gates_per_clifford
    if interleaved and standard_rb_alpha is not None:
        assign_kwargs["standard_rb_alpha"] = standard_rb_alpha

    da = da.assign(**assign_kwargs)

    if interleaved:
        overlay = _try_load_standard_rb_overlay(node, qp_name)
        if overlay is not None:
            da = _assign_standard_rb_overlay(da, overlay)
        else:
            nan_overlay = np.full(len(circuit_depths), np.nan)
            da = da.assign(
                standard_rb_overlay_survival=("circuit_depth", nan_overlay),
                standard_rb_overlay_fitted=("circuit_depth", nan_overlay.copy()),
                standard_rb_fit_alpha=np.nan,
            )

    return da


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Fit RB survival curves for each qubit pair and return an augmented dataset."""
    ds_fit = ds.groupby("qubit_pair").apply(lambda da: fit_rb_routine(da, node))
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Extract RB fit parameters and create FitResults for each qubit pair."""
    qubit_pairs = node.namespace["qubit_pairs"]

    if "survival_probability" in ds_fit.data_vars:
        ds_fit.survival_probability.attrs = {"long_name": "P(|00>)", "units": "a.u."}
    if "survival_per_sequence" in ds_fit.data_vars:
        ds_fit.survival_per_sequence.attrs = {
            "long_name": "P(|00>) per random sequence",
            "units": "a.u.",
        }
    if "fitted_curve" in ds_fit.data_vars:
        ds_fit.fitted_curve.attrs = {"long_name": "exponential RB fit", "units": "a.u."}
    if "fidelity" in ds_fit.data_vars:
        ds_fit.fidelity.attrs = {"long_name": "RB fidelity", "units": "a.u."}
    if "epc" in ds_fit.data_vars:
        ds_fit.epc.attrs = {"long_name": "error per Clifford", "units": "a.u."}
    if "epg" in ds_fit.data_vars:
        ds_fit.epg.attrs = {"long_name": "error per gate", "units": "a.u."}
    if "fit_alpha" in ds_fit.data_vars:
        ds_fit.fit_alpha.attrs = {"long_name": "RB decay constant alpha", "units": "a.u."}

    fit_results: Dict[str, FitResults] = {}
    interleaved = "interleaved" in node.name.lower()
    n_1q = node.namespace.get("avg_1q_per_clifford")
    n_cz = node.namespace.get("avg_cz_per_clifford")
    coherence_limits: list[float] = []
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)
        epc = float(qp_data.epc.values) if "epc" in qp_data else None
        fidelity = float(qp_data.fidelity.values)

        epc_1q_budget = None
        epc_cz_residual = None
        epg_cz_implied = None
        f_1q_control = None
        f_1q_target = None
        if not interleaved and epc is not None and epc > 0:
            f_1q_control = (
                qp.qubit_control.gate_fidelity.get("averaged") if qp.qubit_control.gate_fidelity else None
            )
            f_1q_target = qp.qubit_target.gate_fidelity.get("averaged") if qp.qubit_target.gate_fidelity else None
            if f_1q_control is not None and f_1q_target is not None:
                f_1q_budget = f_1q_control ** (n_1q / 2) * f_1q_target ** (n_1q / 2)
                epc_1q_budget = 1 - f_1q_budget
                epc_cz_residual = epc - epc_1q_budget
                epg_cz_implied = 1 - (fidelity / f_1q_budget) ** (1 / n_cz)

        coherence_limit_epg = try_coherence_limit_epg(qp, node.parameters.operation)
        coherence_limits.append(coherence_limit_epg if coherence_limit_epg is not None else np.nan)

        fit_results[qp_name] = FitResults(
            alpha=float(qp_data.fit_alpha.values),
            fidelity=fidelity,
            success=bool(qp_data.success.values),
            fit_amplitude=float(qp_data.fit_amplitude.values),
            fit_offset=float(qp_data.fit_offset.values),
            epc=epc,
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            average_gate_fidelity=(
                float(qp_data.average_gate_fidelity.values) if "average_gate_fidelity" in qp_data else None
            ),
            average_gates_per_clifford=(
                float(qp_data.average_gates_per_clifford.values)
                if "average_gates_per_clifford" in qp_data
                else None
            ),
            standard_rb_alpha=(
                float(qp_data.standard_rb_alpha.values) if "standard_rb_alpha" in qp_data else None
            ),
            epc_1q_budget=epc_1q_budget,
            epc_cz_residual=epc_cz_residual,
            epg_cz_implied=epg_cz_implied,
            f_1q_control=f_1q_control,
            f_1q_target=f_1q_target,
            avg_1q_per_clifford=n_1q,
            avg_cz_per_clifford=n_cz,
            coherence_limit_epg=coherence_limit_epg,
            fit_issues=(
                tuple(issue for issue in str(qp_data.fit_issues.values).split("\n") if issue)
                if "fit_issues" in qp_data
                else ()
            ),
            fit_warnings=(
                tuple(warning for warning in str(qp_data.fit_warnings.values).split("\n") if warning)
                if "fit_warnings" in qp_data
                else ()
            ),
        )

    ds_fit = ds_fit.assign(
        coherence_limit_epg=("qubit_pair", np.asarray(coherence_limits, dtype=float))
    )
    ds_fit.coherence_limit_epg.attrs = {"long_name": "coherence-limited EPG", "units": "a.u."}

    return ds_fit, fit_results
