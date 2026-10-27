"""RB survival decay ``A * alpha**m + B``: weighted fit and generic fit-quality checks."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import curve_fit

RESIDUAL_SIGMA_LIMIT = 4.0
_FLOAT_TOLERANCE = 1e-9


def rb_decay_curve(x, A, alpha, B):
    """Exponential decay model for RB survival probability."""
    return A * alpha**x + B


def _initial_guess(depths: np.ndarray, survival: np.ndarray) -> list[float]:
    """Start from the data: B at the deepest point, A the drop to it, alpha from the half-way depth."""
    order = np.argsort(depths)
    m, y = depths[order], survival[order]
    offset = float(y[-1])
    amplitude = float(y[0] - offset)
    alpha = 0.9
    if amplitude > 0:
        below = np.nonzero(y - offset < amplitude / 2)[0]
        if below.size and m[below[0]] > 0:
            alpha = float(0.5 ** (1 / m[below[0]]))
    return [amplitude, alpha, offset]


def fit_survival(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Fit ``A * alpha**m + B``, weighted by ``stderr`` when every point has one.

    Returns ``(A, alpha, B, alpha_stderr, A_stderr, B_stderr)``, or None if the fit fails.
    """
    depths = np.asarray(circuit_depths, dtype=float)
    values = np.asarray(survival, dtype=float)
    sigma = None
    if stderr is not None:
        err = np.asarray(stderr, dtype=float)
        if err.shape == values.shape and np.all(np.isfinite(err)) and np.all(err > 0):
            sigma = err

    try:
        popt, pcov = curve_fit(
            rb_decay_curve,
            depths,
            values,
            p0=_initial_guess(depths, values),
            maxfev=10000,
            sigma=sigma,
            absolute_sigma=sigma is not None,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("RB exponential fit failed: %s", exc)
        return None

    stderrs = [np.nan, np.nan, np.nan]
    if pcov is not None and pcov.shape == (3, 3):
        for i in range(3):
            if np.isfinite(pcov[i, i]) and pcov[i, i] >= 0:
                stderrs[i] = float(np.sqrt(pcov[i, i]))
    amplitude_stderr, alpha_stderr, offset_stderr = stderrs
    return float(popt[0]), float(popt[1]), float(popt[2]), alpha_stderr, amplitude_stderr, offset_stderr


def validate_fit_parameters(
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
    *,
    probability: bool = True,
    amplitude_stderr: float | None = None,
    offset_stderr: float | None = None,
) -> list[str]:
    """Hard-fail checks on the fitted coefficients.

    ``probability``: the survival is a probability, so A >= 0, B >= 0 and A + B <= 1 also apply
    (not for a raw I/Q signal). With the A and B errors, A + B may exceed 1 by up to 3 sigma: with
    near-perfect readout the fitted start of the curve can land just above 1 by noise.
    """
    issues: list[str] = []
    if not all(np.isfinite(v) for v in (fit_amplitude, alpha, fit_offset)):
        issues.append("Non-finite fit parameters (A, alpha, or B).")
        return issues

    if alpha <= _FLOAT_TOLERANCE or alpha > 1.0 + _FLOAT_TOLERANCE:
        issues.append(f"RB decay alpha={alpha:.6f} outside physical range (0, 1].")
    if not probability:
        return issues

    if fit_amplitude < -_FLOAT_TOLERANCE:
        issues.append(f"Fit amplitude A={fit_amplitude:.6f} must be >= 0.")

    if fit_offset < -_FLOAT_TOLERANCE:
        issues.append(f"Fit offset B={fit_offset:.6f} must be >= 0.")

    tolerance = _FLOAT_TOLERANCE
    if amplitude_stderr is not None and offset_stderr is not None:
        sum_stderr = np.hypot(amplitude_stderr, offset_stderr)
        if np.isfinite(sum_stderr):
            tolerance += 3 * sum_stderr
    if fit_amplitude + fit_offset > 1.0 + tolerance:
        issues.append(f"A + B = {fit_amplitude + fit_offset:.6f} exceeds 1 (invalid survival-probability model).")
    return issues


def validate_residuals(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
) -> list[str]:
    """Flag a fitted curve more than ``RESIDUAL_SIGMA_LIMIT`` standard errors from the data."""
    fitted_values = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    safe_stderr = np.where(stderr > 0, stderr, np.nan)
    normalized_residuals = np.abs(fitted_values - survival) / safe_stderr
    if not np.isfinite(normalized_residuals).any():
        return ["Cannot assess fit residuals (zero or missing standard errors)."]

    max_deviation = float(np.nanmax(normalized_residuals))
    if max_deviation > RESIDUAL_SIGMA_LIMIT:
        return [
            f"Fitted curve deviates up to {max_deviation:.2f} sigma from experimental data "
            f"(limit {RESIDUAL_SIGMA_LIMIT:.0f} sigma)."
        ]
    return []
