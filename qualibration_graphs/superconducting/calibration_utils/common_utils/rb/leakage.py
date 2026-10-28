"""Leakage RB model and fit, shared by the single- and two-qubit RB nodes.

Leaked population after m Cliffords with leakage L1 and seepage L2 per Clifford (Wood &
Gambetta, PRA 97, 032306 (2018)):  P_L(m) = p_inf + (p0 - p_inf) * lam**m,  lam = 1 - L1 - L2,
p_inf = L1 / (L1 + L2). Small leakage rarely saturates at reachable depths: p_inf and lam are
then loose while their product, the initial slope L1, is not. A fit that cannot pin L1 falls
back to a straight line whose slope is L1.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import xarray as xr
from scipy.optimize import OptimizeWarning, curve_fit

RESIDUAL_SIGMA_LIMIT = 4.0
# Empirical, like the RB survival soft checks: tolerances that flag clearly-off data without
# false-positiving on ordinary shot noise.
MONOTONIC_TOLERANCE = 0.03
# A leaked population this high at the shallowest depth is readout or reset, not leakage.
SHALLOW_LEAKAGE_THRESHOLD = 0.1


@dataclass
class LeakageFit:
    """Leakage RB fit (``measure_leakage=True``): leakage ``l1`` and seepage ``l2`` per Clifford.

    ``method`` is ``exponential`` (full model) or ``linear`` (slope only, ``l2`` unknown). Like the
    survival fits, ``fit_issues`` says why a fit was rejected and ``fit_warnings`` holds soft
    warnings that do not fail it.
    """

    l1: float | None = float("nan")
    l1_stderr: float | None = float("nan")
    l2: float | None = float("nan")
    l2_stderr: float | None = float("nan")
    p_inf: float | None = float("nan")
    lam: float | None = float("nan")
    p0: float | None = float("nan")
    method: str = ""
    success: bool = False
    fit_issues: tuple[str, ...] = field(default_factory=tuple)
    fit_warnings: tuple[str, ...] = field(default_factory=tuple)

    def curve(self, depths) -> np.ndarray:
        """Fitted leaked population at ``depths``."""
        m = np.asarray(depths, dtype=float)
        if not self.success:
            return np.full_like(m, np.nan)
        if self.method == "linear":
            return self.p0 + self.l1 * m
        return self.p_inf + (self.p0 - self.p_inf) * self.lam**m


def leakage_decay_curve(x, p_inf, lam, p0):
    """Leaked population after ``x`` Cliffords (leakage RB model, see the module docstring)."""
    return p_inf + (p0 - p_inf) * lam**x


def _check_leakage_soft_warnings(depths: np.ndarray, leaked: np.ndarray, stderr: np.ndarray | None) -> list[str]:
    """Soft checks on raw leaked-population data (warnings only; do not fail the fit)."""
    found: list[str] = []
    shallow_idx, deep_idx = int(np.argmin(depths)), int(np.argmax(depths))
    shallow_depth, deep_depth = float(depths[shallow_idx]), float(depths[deep_idx])
    shallow, deep = float(leaked[shallow_idx]), float(leaked[deep_idx])

    if shallow > SHALLOW_LEAKAGE_THRESHOLD:
        found.append(
            f"Leaked population at depth {shallow_depth:g} is {shallow:.3f} (above "
            f"{SHALLOW_LEAKAGE_THRESHOLD:g}) — at the shallowest depth this is g/e/f readout or reset, "
            "not leakage; check gef_centers (15_iq_blobs_gef). Fitted leakage may be unreliable."
        )
    if deep_depth > shallow_depth:
        err = np.zeros_like(leaked) if stderr is None else stderr
        tolerance = max(float(err[shallow_idx]), float(err[deep_idx]), MONOTONIC_TOLERANCE)
        if deep < shallow - tolerance:
            found.append(
                f"Leaked population falls from {shallow:.3f} at depth {shallow_depth:g} to {deep:.3f} "
                f"at depth {deep_depth:g} — not a leakage build-up; check g/e/f readout."
            )
    return found


def _validate_leakage_residuals(
    depths: np.ndarray, leaked: np.ndarray, stderr: np.ndarray | None, fit: LeakageFit
) -> list[str]:
    """Hard-fail when the fitted curve deviates too far from the data (as for the survival fit)."""
    if stderr is None:
        return []
    safe_stderr = np.where(stderr > 0, stderr, np.nan)
    normalized_residuals = np.abs(fit.curve(depths) - leaked) / safe_stderr
    if not np.isfinite(normalized_residuals).any():
        return ["Cannot assess fit residuals (zero or missing standard errors)."]
    max_deviation = float(np.nanmax(normalized_residuals))
    if max_deviation > RESIDUAL_SIGMA_LIMIT:
        return [
            f"Fitted {fit.method} curve deviates up to {max_deviation:.2f} sigma from the data "
            f"(limit {RESIDUAL_SIGMA_LIMIT:.0f} sigma)."
        ]
    return []


def _fit_leakage_exponential(m: np.ndarray, p: np.ndarray, sigma: np.ndarray | None) -> LeakageFit:
    """Full model; ``success=False`` with ``fit_issues`` when it does not converge or pin L1."""
    slope = max(float((p[-1] - p[0]) / ((m[-1] - m[0]) or 1.0)), 1e-6)
    p_inf_guess = float(np.clip(max(p[-1] * 2.0, slope * 100.0), 1e-4, 0.99))
    guess = [p_inf_guess, float(np.clip(1.0 - slope / p_inf_guess, 0.5, 1.0 - 1e-6)), float(np.clip(p[0], 0, 1))]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            (p_inf, lam, p0), cov = curve_fit(
                leakage_decay_curve,
                m,
                p,
                p0=guess,
                sigma=sigma,
                absolute_sigma=sigma is not None,
                bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("Leakage exponential fit failed: %s", exc)
        return LeakageFit(method="exponential", fit_issues=(f"Exponential fit failed: {exc}",))
    if not np.all(np.isfinite(cov)):
        return LeakageFit(method="exponential", fit_issues=("Exponential fit has no usable covariance.",))

    l1, l2 = p_inf * (1.0 - lam), (1.0 - p_inf) * (1.0 - lam)
    grad_l1 = np.array([1.0 - lam, -p_inf, 0.0])
    grad_l2 = np.array([-(1.0 - lam), -(1.0 - p_inf), 0.0])
    l1_stderr = float(np.sqrt(max(grad_l1 @ cov @ grad_l1, 0.0)))
    fit = LeakageFit(
        l1=float(l1),
        l1_stderr=l1_stderr,
        l2=float(l2),
        l2_stderr=float(np.sqrt(max(grad_l2 @ cov @ grad_l2, 0.0))),
        p_inf=float(p_inf),
        lam=float(lam),
        p0=float(p0),
        method="exponential",
        success=True,
    )
    if l1 > 0 and l1_stderr > l1:
        fit.success = False
        fit.fit_issues = (f"Exponential fit leaves L1 unconstrained ({l1:.2e} +/- {l1_stderr:.2e}).",)
    return fit


def _fit_leakage_linear(m: np.ndarray, p: np.ndarray, sigma: np.ndarray | None) -> LeakageFit:
    """Straight line: the slope is L1, L2 unknown (leakage not saturated)."""
    try:
        coeffs, cov = np.polyfit(
            m,
            p,
            1,
            w=None if sigma is None else 1.0 / sigma,
            cov="unscaled" if sigma is not None else True,
        )
    except (np.linalg.LinAlgError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("Leakage linear fit failed: %s", exc)
        return LeakageFit(method="linear", fit_issues=(f"Linear fit failed: {exc}",))
    slope, intercept = (float(x) for x in coeffs)
    return LeakageFit(
        l1=slope,
        l1_stderr=float(np.sqrt(max(cov[0, 0], 0.0))),
        p0=intercept,
        method="linear",
        success=bool(np.isfinite(slope)),
    )


def fit_leakage(depths, leaked, stderr=None) -> LeakageFit:
    """Fit leaked population against depth: the exponential model, else a straight line.

    A fit is kept only if it converges, pins down L1 and stays within the residual limit; the
    exponential is tried first. Soft warnings on the data (``fit_warnings``) never fail the fit.
    """
    m = np.asarray(depths, dtype=float)
    p = np.asarray(leaked, dtype=float)
    sigma = None if stderr is None else np.asarray(stderr, dtype=float)
    keep = np.isfinite(m) & np.isfinite(p) & (True if sigma is None else np.isfinite(sigma) & (sigma > 0))
    m, p = m[keep], p[keep]
    sigma = None if sigma is None else sigma[keep]
    if m.size < 3:
        return LeakageFit(fit_issues=(f"Need at least 3 depths, got {m.size}.",))
    order = np.argsort(m)
    m, p = m[order], p[order]
    sigma = None if sigma is None else sigma[order]
    soft_warnings = tuple(_check_leakage_soft_warnings(m, p, sigma))

    issues: list[str] = []
    for fitter in (_fit_leakage_exponential, _fit_leakage_linear):
        fit = fitter(m, p, sigma)
        if fit.success:
            residual_issues = _validate_leakage_residuals(m, p, sigma, fit)
            if not residual_issues:
                # A rejected exponential before an accepted line is a warning, not a fit issue.
                fallback = ("Linear slope only (leakage not saturated): L1 from the slope, L2 unknown.",)
                fit.fit_warnings = soft_warnings + tuple(issues) + (fallback if fit.method == "linear" else ())
                return fit
            fit.success = False
            fit.fit_issues = tuple(residual_issues)
        issues.extend(fit.fit_issues)
    return LeakageFit(method=fit.method, fit_issues=tuple(issues), fit_warnings=soft_warnings)


def leakage_per_gate(fits: Sequence[LeakageFit], references: Sequence[Optional[dict]]) -> dict[str, list[float]]:
    """Leakage per interleaved gate of each series: interleaved minus reference, errors in quadrature.

    ``fits`` are the interleaved run's fits and ``references`` the reference run's stored fits of
    the same series, in the same order (dicts with ``l1``, ``l1_stderr``, ``l2``, ``l2_stderr``;
    ``None`` or empty when missing). Interleaving the gate adds its leakage to every step, so for
    small rates the gate's leakage is the difference of the two per-step rates. L2 only when both
    fits have it; a missing value is NaN.

    Returns:
        Lists keyed ``l1``, ``l1_stderr``, ``l2`` and ``l2_stderr``, one entry per series.
    """
    out = {"l1": [], "l1_stderr": [], "l2": [], "l2_stderr": []}
    for fit, ref in zip(fits, references):
        ref = ref or {}
        l1 = l1_err = l2 = l2_err = np.nan
        if fit.success and ref.get("l1") is not None:
            l1 = fit.l1 - ref["l1"]
            l1_err = float(np.hypot(fit.l1_stderr, ref.get("l1_stderr") or 0.0))
            if np.isfinite(fit.l2) and ref.get("l2") is not None:
                l2 = fit.l2 - ref["l2"]
                l2_err = float(np.hypot(fit.l2_stderr, ref.get("l2_stderr") or 0.0))
        for key, value in zip(out, (l1, l1_err, l2, l2_err)):
            out[key].append(value)
    return out


def merge_leakage(ds_fit: xr.Dataset, ds_leakage: xr.Dataset) -> xr.Dataset:
    """``ds_fit`` with the leakage fit added, so a run saves one dataset.

    Leakage variables are named ``leaked*`` / ``leakage_*``; a variable already in ``ds_fit`` (the
    raw data both were built from) must hold the same values and is kept once.
    """
    shared = [name for name in ds_leakage.data_vars if name in ds_fit.data_vars]
    clashes = [name for name in shared if not ds_leakage[name].equals(ds_fit[name])]
    if clashes:
        raise ValueError(f"Leakage and survival fits both define {clashes} with different values.")
    return ds_fit.merge(ds_leakage.drop_vars(shared))
