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
     Implied CZ EPG is inferred from Eq. (10) of supplemental material of McKay et al.
     (https://arxiv.org/abs/1712.06550v2) using 1Q RB ``alpha`` values and
     transpiled gate counts (use 37b for a direct CZ measurement).
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
   (``StandardRB_load_id``) and attach its saved ``ds_fit`` curves and ``alpha`` ± for plotting.

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
    alpha_stderr: float | None = None
    fidelity_stderr: float | None = None
    epc: float | None = None
    epg: float | None = None
    epg_stderr: float | None = None
    average_gate_fidelity: float | None = None
    average_gates_per_clifford: float | None = None
    standard_rb_alpha: float | None = None
    standard_rb_alpha_stderr: float | None = None
    alpha_0: float | None = None
    alpha_1: float | None = None
    alpha_01_implied: float | None = None
    alpha_01_stderr: float | None = None
    epg_cz_implied: float | None = None
    epg_cz_implied_stderr: float | None = None
    avg_1q_per_clifford: float | None = None
    avg_cz_per_clifford: float | None = None
    coherence_limit_epg: float | None = None
    fit_issues: tuple[str, ...] = field(default_factory=tuple)
    fit_warnings: tuple[str, ...] = field(default_factory=tuple)


def format_fraction_pm(
    value: float | None,
    stderr: float | None = None,
    *,
    as_error_rate: bool = False,
) -> str:
    """Format a fraction in [0, 1] as a percentage, optionally with ±1σ.

    When *as_error_rate* is True, values below 1% use ``× 10⁻³`` notation
    (with or without uncertainty).
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    if stderr is None or not np.isfinite(stderr) or stderr <= 0:
        if as_error_rate and value < 0.01:
            return f"{value * 1e3:.2f} × 10⁻³"
        return f"{100 * value:.2f}%"
    if as_error_rate and value < 0.01:
        return f"{value * 1e3:.2f} ± {stderr * 1e3:.2f} × 10⁻³"
    return f"{100 * value:.2f} ± {100 * stderr:.2f}%"


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
            alpha_label = "alpha_IRB" if interleaved else "alpha"
            alpha_text = (
                f"{fit_result.alpha:.6f} ± {fit_result.alpha_stderr:.6f}"
                if fit_result.alpha_stderr is not None
                and np.isfinite(fit_result.alpha_stderr)
                and fit_result.alpha_stderr > 0
                else f"{fit_result.alpha:.6f}"
            )
            lines.append(
                f"\tDecay model: P(m) = A * alpha^m + B "
                f"(A = {fit_result.fit_amplitude:.6f}, {alpha_label} = {alpha_text}, "
                f"B = {fit_result.fit_offset:.6f})"
            )
            if interleaved:
                alpha_srb_text = (
                    f"{fit_result.standard_rb_alpha:.6f} ± {fit_result.standard_rb_alpha_stderr:.6f}"
                    if fit_result.standard_rb_alpha is not None
                    and fit_result.standard_rb_alpha_stderr is not None
                    and np.isfinite(fit_result.standard_rb_alpha_stderr)
                    and fit_result.standard_rb_alpha_stderr > 0
                    else (
                        f"{fit_result.standard_rb_alpha:.6f}"
                        if fit_result.standard_rb_alpha is not None
                        else "n/a"
                    )
                )
                srb_fidelity = 1 - fit_result.epc if fit_result.epc is not None else np.nan
                srb_fidelity_stderr = (
                    _fidelity_stderr_from_alpha(fit_result.standard_rb_alpha_stderr, interleaved=False)
                    if fit_result.standard_rb_alpha_stderr is not None
                    and np.isfinite(fit_result.standard_rb_alpha_stderr)
                    and fit_result.standard_rb_alpha_stderr > 0
                    else None
                )
                lines.extend(
                    [
                        "",
                        "\tStandard RB reference (37a):",
                        f"\t2Q Clifford Fidelity = "
                        f"{format_fraction_pm(srb_fidelity, srb_fidelity_stderr)}",
                        f"\tError Per Clifford (EPC) = 1 - 2Q Clifford Fidelity = "
                        f"{format_fraction_pm(fit_result.epc, srb_fidelity_stderr, as_error_rate=True)}",
                        f"\talpha_SRB = {alpha_srb_text}",
                        "",
                        "\tInterleaved RB (this run):",
                        f"\tCZ gate fidelity = 1 - (d-1)/d * (1 - alpha_IRB/alpha_SRB) = "
                        f"{format_fraction_pm(fit_result.fidelity, fit_result.fidelity_stderr)}",
                        f"\talpha_IRB = {alpha_text}",
                        f"\tError Per Gate (EPG) = 1 - CZ gate fidelity = "
                        f"{format_fraction_pm(fit_result.epg, fit_result.epg_stderr, as_error_rate=True)}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"\t2Q Clifford Fidelity = "
                        f"{format_fraction_pm(fit_result.fidelity, fit_result.fidelity_stderr)}",
                        f"\tError Per Clifford (EPC): 1 - 2Q Clifford Fidelity = "
                        f"{format_fraction_pm(fit_result.epc, fit_result.fidelity_stderr, as_error_rate=True)}",
                        f"\tError Per Gate (EPG) = EPC / N_gates_per_Clifford = "
                        f"{format_fraction_pm(fit_result.epc, as_error_rate=True)} / {fit_result.average_gates_per_clifford:.2f} = "
                        f"{format_fraction_pm(fit_result.epg, as_error_rate=True)}",
                        f"\tAvg. Gate Fidelity (1-EPG) = {100 * fit_result.average_gate_fidelity:.2f}%",
                    ]
                )

            if not interleaved and fit_result.alpha_01_implied is not None:
                alpha_2q_text = (
                    f"{fit_result.alpha:.6f} ± {fit_result.alpha_stderr:.6f}"
                    if fit_result.alpha_stderr is not None
                    and np.isfinite(fit_result.alpha_stderr)
                    and fit_result.alpha_stderr > 0
                    else f"{fit_result.alpha:.6f}"
                )
                alpha_01_text = (
                    f"{fit_result.alpha_01_implied:.6f} ± {fit_result.alpha_01_stderr:.6f}"
                    if fit_result.alpha_01_stderr is not None
                    and np.isfinite(fit_result.alpha_01_stderr)
                    and fit_result.alpha_01_stderr > 0
                    else f"{fit_result.alpha_01_implied:.6f}"
                )
                lines.extend(
                    [
                        "",
                        "\tImplied CZ gate fidelity:",
                        f"\talpha_2Q = {alpha_2q_text} (measured in this run), "
                        f"alpha_01 = {alpha_01_text} (inferred by factoring out 1Q contributions)",
                        f"\tImplied CZ Error Per Gate (EPG) = "
                        f"{format_fraction_pm(fit_result.epg_cz_implied, fit_result.epg_cz_implied_stderr, as_error_rate=True)}",
                        f"\tImplied CZ gate fidelity = 1 - EPG = "
                        f"{format_fraction_pm(None if fit_result.epg_cz_implied is None else 1 - fit_result.epg_cz_implied, fit_result.epg_cz_implied_stderr)}",
                        "\tNote: Interleaved RB (37b) measures CZ EPG directly.",
                    ]
                )

            if fit_result.coherence_limit_epg is not None:
                lines.append(
                    f"\tCoherence-limited EPG (T1/T2 floor) = "
                    f"{format_fraction_pm(fit_result.coherence_limit_epg, as_error_rate=True)}"
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
                        f"({ratio:.1f}× floor: {format_fraction_pm(fit_result.epg, fit_result.epg_stderr, as_error_rate=True)} vs "
                        f"{format_fraction_pm(fit_result.coherence_limit_epg, as_error_rate=True)})"
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


def _infer_alpha_01(
    alpha_2q: float,
    qubit_control,
    qubit_target,
    n_1: float,
    n_2: float,
) -> tuple[float | None, float | None, float | None]:
    """Infer per-CZ RB decay alpha_01 by inverting Eq. (10) of McKay et al.

    See https://arxiv.org/abs/1712.06550v2 (Phys. Rev. Lett. 122, 200502).

    Returns ``(alpha_0, alpha_1, alpha_01)``; components are ``None`` when inputs are missing
    or the inversion is ill-defined.
    """
    if n_2 <= 0 or not np.isfinite(n_1) or n_1 < 0:
        return None, None, None

    def _qubit_rb_alpha(qubit) -> float | None:
        gf = getattr(qubit, "gate_fidelity", None) or {}
        if "averaged" not in gf:
            return None
        f = float(gf["averaged"])
        if not np.isfinite(f):
            return None
        # 1Q RB stores gate fidelity; McKay Eq. (10) needs alpha: f = (1 + alpha) / 2 for d = 2.
        return 2.0 * f - 1.0

    alpha_0 = _qubit_rb_alpha(qubit_control)
    alpha_1 = _qubit_rb_alpha(qubit_target)
    if alpha_0 is None or alpha_1 is None:
        return alpha_0, alpha_1, None
    if not np.isfinite(alpha_2q) or alpha_2q <= 0 or alpha_0 <= 0 or alpha_1 <= 0:
        return alpha_0, alpha_1, None

    half = n_1 / 2.0
    a0, a1 = alpha_0**half, alpha_1**half
    factor_1q = (a0 + a1 + 3.0 * a0 * a1) / 5.0
    if factor_1q <= 0:
        return alpha_0, alpha_1, None

    ratio = alpha_2q / factor_1q
    if ratio <= 0:
        return alpha_0, alpha_1, None

    return alpha_0, alpha_1, float(ratio ** (1.0 / n_2))


def _alpha_01_stderr_from_alpha_2q(
    alpha_01: float,
    alpha_2q: float,
    alpha_2q_stderr: float | None,
    n_2: float,
) -> float | None:
    """Propagate fit uncertainty on ``alpha_2q`` to inferred ``alpha_01`` (1Q inputs held fixed).

    For ``alpha_01 = (alpha_2q / factor_1q)^(1 / n_2)``, d(alpha_01)/d(alpha_2q) = alpha_01 / (n_2 * alpha_2q).
    """
    if (
        alpha_2q_stderr is None
        or not np.isfinite(alpha_2q_stderr)
        or alpha_2q_stderr <= 0
        or not np.isfinite(alpha_01)
        or not np.isfinite(alpha_2q)
        or alpha_2q <= 0
        or n_2 <= 0
    ):
        return None
    return alpha_01 * alpha_2q_stderr / (n_2 * alpha_2q)


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


def _fidelity_stderr_from_alpha(
    alpha_stderr: float,
    *,
    interleaved: bool,
    standard_rb_alpha: float | None = None,
    n_qubits: int = 2,
) -> float:
    """Propagate 1σ uncertainty on RB decay ``alpha`` to fidelity (1σ)."""
    if not np.isfinite(alpha_stderr) or alpha_stderr <= 0:
        return np.nan

    d = 2**n_qubits
    if interleaved:
        if standard_rb_alpha is None or not np.isfinite(standard_rb_alpha) or standard_rb_alpha <= 0:
            return np.nan
        return ((d - 1) / d) * alpha_stderr / standard_rb_alpha

    return ((d - 1) / d) * alpha_stderr


def _fit_survival(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    """Fit ``A * alpha**m + B`` and return ``(A, alpha, B, alpha_stderr)``."""
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

    alpha_stderr = np.nan
    if pcov is not None and pcov.shape == (3, 3) and np.isfinite(pcov[1, 1]) and pcov[1, 1] >= 0:
        alpha_stderr = float(np.sqrt(pcov[1, 1]))

    return float(popt[0]), float(popt[1]), float(popt[2]), alpha_stderr


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
    """Load saved 37a ``ds_fit`` for interleaved overlay plots (no refit)."""
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

        survival_per_sequence = fr.survival_per_sequence if "survival_per_sequence" in fr else None

        return {
            "circuit_depth": np.asarray(fr.circuit_depth.values, dtype=float),
            "survival": np.asarray(fr.survival_probability.values, dtype=float),
            "survival_per_sequence": survival_per_sequence,
            "fitted_curve": np.asarray(fr.fitted_curve.values, dtype=float),
            "alpha": float(fr.fit_alpha.values),
            "alpha_stderr": alpha_stderr,
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
        "standard_rb_fit_alpha_stderr": float(overlay.get("alpha_stderr", np.nan)),
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

    fit_params = _fit_survival(circuit_depths, survival_vals, stderr_vals)
    if fit_params is None:
        fit_amplitude = np.nan
        alpha = np.nan
        fit_offset = np.nan
        alpha_stderr = np.nan
        fidelity_stderr = np.nan
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
        fit_amplitude, alpha, fit_offset, alpha_stderr = fit_params
        standard_rb_alpha = None
        fidelity_stderr = np.nan

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

        fidelity_stderr = _fidelity_stderr_from_alpha(
            alpha_stderr,
            interleaved=interleaved,
            standard_rb_alpha=standard_rb_alpha,
        )

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
        "alpha_stderr": alpha_stderr,
        "fidelity": fidelity,
        "fidelity_stderr": fidelity_stderr,
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
                standard_rb_fit_alpha_stderr=np.nan,
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
        fidelity_stderr = (
            float(qp_data.fidelity_stderr.values)
            if "fidelity_stderr" in qp_data and np.isfinite(qp_data.fidelity_stderr.values)
            else None
        )
        epg_stderr = fidelity_stderr if interleaved else None

        standard_rb_alpha = (
            float(qp_data.standard_rb_alpha.values) if interleaved and "standard_rb_alpha" in qp_data else None
        )
        standard_rb_alpha_stderr = None
        if interleaved and "standard_rb_fit_alpha_stderr" in qp_data:
            stderr = float(qp_data.standard_rb_fit_alpha_stderr.values)
            if np.isfinite(stderr) and stderr > 0:
                standard_rb_alpha_stderr = stderr

        alpha_0 = None
        alpha_1 = None
        alpha_01_implied = None
        alpha_01_stderr = None
        epg_cz_implied = None
        epg_cz_implied_stderr = None
        if not interleaved and n_1q is not None and n_cz is not None:
            alpha_2q = float(qp_data.fit_alpha.values)
            alpha_2q_stderr = (
                float(qp_data.alpha_stderr.values)
                if "alpha_stderr" in qp_data and np.isfinite(qp_data.alpha_stderr.values)
                else None
            )
            alpha_0, alpha_1, alpha_01_implied = _infer_alpha_01(
                alpha_2q,
                qp.qubit_control,
                qp.qubit_target,
                n_1q,
                n_cz,
            )
            if alpha_01_implied is not None:
                alpha_01_stderr = _alpha_01_stderr_from_alpha_2q(alpha_01_implied, alpha_2q, alpha_2q_stderr, n_cz)
                epg_cz_implied = 1.0 - clifford_fidelity_from_alpha(alpha_01_implied, n_qubits=2)
                if alpha_01_stderr is not None:
                    epg_cz_implied_stderr = _fidelity_stderr_from_alpha(alpha_01_stderr, interleaved=False, n_qubits=2)

        coherence_limit_epg = try_coherence_limit_epg(qp, node.parameters.operation)
        coherence_limits.append(coherence_limit_epg if coherence_limit_epg is not None else np.nan)

        fit_results[qp_name] = FitResults(
            alpha=float(qp_data.fit_alpha.values),
            fidelity=fidelity,
            success=bool(qp_data.success.values),
            fit_amplitude=float(qp_data.fit_amplitude.values),
            fit_offset=float(qp_data.fit_offset.values),
            alpha_stderr=(float(qp_data.alpha_stderr.values) if "alpha_stderr" in qp_data else None),
            fidelity_stderr=fidelity_stderr,
            epc=epc,
            epg=float(qp_data.epg.values) if "epg" in qp_data else None,
            epg_stderr=epg_stderr,
            average_gate_fidelity=(
                float(qp_data.average_gate_fidelity.values) if "average_gate_fidelity" in qp_data else None
            ),
            average_gates_per_clifford=(
                float(qp_data.average_gates_per_clifford.values) if "average_gates_per_clifford" in qp_data else None
            ),
            standard_rb_alpha=standard_rb_alpha,
            standard_rb_alpha_stderr=standard_rb_alpha_stderr,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            alpha_01_implied=alpha_01_implied,
            alpha_01_stderr=alpha_01_stderr,
            epg_cz_implied=epg_cz_implied,
            epg_cz_implied_stderr=epg_cz_implied_stderr,
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

    ds_fit = ds_fit.assign(coherence_limit_epg=("qubit_pair", np.asarray(coherence_limits, dtype=float)))
    ds_fit.coherence_limit_epg.attrs = {"long_name": "coherence-limited EPG", "units": "a.u."}

    return ds_fit, fit_results
