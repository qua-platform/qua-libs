"""
Robust sensor compensation via row-wise Lorentzian peak tracking + BayesianCP.
==============================================================================

Extracts the cross-talk gradient *alpha* from a 2D sensor-vs-device scan
that may contain charge transitions along the device axis.

Pipeline
--------
1. **Global shape estimation**: fit the full 2D scan to a shifted Lorentzian
   to obtain shared amplitude *A*, width *gamma*, and *offset*.

2. **Row-wise peak extraction**: for each device-gate row, fit only the
   peak position (single parameter) with shape parameters fixed.

3. **Profile-likelihood alpha estimation**: find the alpha that maximises
   the BayesianCP log-evidence on the detrended peak trace
   ``peaks - alpha * v_d``.  Because the BCP segment model is
   piecewise-constant, the optimal alpha is the one that makes the
   residuals most piecewise-constant — coupling gradient estimation and
   change-point detection without a biased initial estimate.

4. **Final BCP + least-squares**: at the optimal alpha, run BCP once more
   to extract change points and fit per-segment intercepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .bayesian_cp import BayesianCP


@dataclass
class RobustSensorCompResult:
    """Result from the robust peak-tracking + BCP pipeline."""

    alpha: float
    alpha_std: float
    peak_positions: np.ndarray
    changepoint_indices: List[int]
    segment_intercepts: List[float]
    n_changepoints: int
    A: float
    gamma: float
    offset: float
    cp_posterior: Optional[np.ndarray] = field(default=None, repr=False)


def fit_robust_sensor_compensation(
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    signal: np.ndarray,
    *,
    hazard: float = 1 / 30.0,
    cp_threshold: float = 0.3,
    initial_fit: Optional[Dict[str, float]] = None,
) -> RobustSensorCompResult:
    """
    Extract the cross-talk gradient alpha, robust to charge transitions.

    Instead of estimating alpha first (which is biased by charge
    transitions) and then running change-point detection, this function
    **jointly** determines alpha and the change points by maximising the
    BayesianCP log-evidence over alpha:

        alpha* = argmax_alpha  log p(peaks - alpha * v_d | BCP model)

    For each candidate alpha the detrended peak trace should be
    piecewise-constant (with jumps only at charge transitions), so the
    BCP evidence naturally scores how well alpha separates the linear
    trend from the step-like transitions.

    Parameters
    ----------
    v_sensor : (M,) array
        Sensor gate voltages.
    v_device : (N,) array
        Device gate voltages (rows of the 2D scan).
    signal : (N, M) array
        Measured 2D signal.
    hazard : float
        Hazard rate for BayesianCP (smaller = fewer expected CPs).
    cp_threshold : float
        Posterior probability threshold above which a point is marked as
        a change point.
    initial_fit : dict, optional
        Pre-computed global fit parameters (``A``, ``v0``, ``alpha``,
        ``gamma``, ``offset``).  When *None*, estimated from the data.

    Returns
    -------
    RobustSensorCompResult
    """
    import jax.numpy as jnp

    v_s = np.asarray(v_sensor, dtype=np.float64)
    v_d = np.asarray(v_device, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)

    # Step 1: global shape parameters
    if initial_fit is not None:
        A, gamma, offset = initial_fit["A"], initial_fit["gamma"], initial_fit["offset"]
    else:
        A, gamma, offset = _estimate_shape(v_s, sig)

    # Step 2: per-row peak position
    peak_pos = _extract_peak_positions(v_s, sig, A, gamma, offset)

    # Identify rows where the peak is reliably inside the scan window
    margin = (v_s[-1] - v_s[0]) * 0.02
    interior = (peak_pos > v_s[0] + margin) & (peak_pos < v_s[-1] - margin)
    if np.sum(interior) < 10:
        interior = np.ones(len(v_d), dtype=bool)

    v_d_int = v_d[interior]
    peak_int = peak_pos[interior]

    # Step 3: maximise BCP log-evidence over alpha
    # The evidence landscape can be non-convex, so we use a coarse grid
    # scan to find the right basin then refine with bounded Brent.
    vs_span = float(v_s[-1] - v_s[0])
    vd_span = float(v_d[-1] - v_d[0]) if abs(v_d[-1] - v_d[0]) > 1e-15 else 1.0
    alpha_max = 3.0 * vs_span / abs(vd_span)

    def _bcp_log_evidence(alpha_candidate: float) -> float:
        residual = peak_int - alpha_candidate * v_d_int
        bcp = BayesianCP(hazard=hazard, standardize=True)
        _, log_ev = bcp.fit(jnp.asarray(residual))
        return float(log_ev)

    n_grid = 40
    grid = np.linspace(-alpha_max, alpha_max, n_grid)
    ev_grid = np.array([_bcp_log_evidence(a) for a in grid])
    best_grid_idx = int(np.argmax(ev_grid))

    # Refine around the best grid point
    lo = float(grid[max(best_grid_idx - 1, 0)])
    hi = float(grid[min(best_grid_idx + 1, n_grid - 1)])
    opt = minimize_scalar(
        lambda a: -_bcp_log_evidence(a),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6 * alpha_max},
    )
    alpha_opt = float(opt.x)

    # Step 4: final BCP run at optimal alpha to get posteriors + CPs
    residual_opt = peak_int - alpha_opt * v_d_int
    cp_final = BayesianCP(hazard=hazard, standardize=True)
    cp_posterior, _ = cp_final.fit(jnp.asarray(residual_opt))
    cp_posterior = np.asarray(cp_posterior)

    cp_indices_local = list(np.where(cp_posterior > cp_threshold)[0] + 1)
    cp_indices_local = _merge_nearby_cps(cp_indices_local, min_gap=3)

    full_indices = np.where(interior)[0]
    cp_indices = [int(full_indices[i]) for i in cp_indices_local if i < len(full_indices)]

    # Step 5: per-segment intercepts + alpha uncertainty via least-squares
    alpha, alpha_std, intercepts = _fit_piecewise_linear(
        v_d_int,
        peak_int,
        cp_indices_local,
    )

    return RobustSensorCompResult(
        alpha=alpha,
        alpha_std=alpha_std,
        peak_positions=peak_pos,
        changepoint_indices=cp_indices,
        segment_intercepts=intercepts,
        n_changepoints=len(cp_indices),
        A=A,
        gamma=gamma,
        offset=offset,
        cp_posterior=cp_posterior,
    )


def _estimate_shape(v_s: np.ndarray, sig: np.ndarray) -> tuple[float, float, float]:
    """Estimate A, gamma, offset from the column-averaged 1D profile."""
    avg_profile = np.mean(sig, axis=0)
    offset = float(np.min(avg_profile))
    A = float(np.max(avg_profile) - offset)
    peak_idx = int(np.argmax(avg_profile))
    half_max = offset + A / 2.0
    above = avg_profile > half_max
    if np.any(above):
        left = np.argmax(above)
        right = len(above) - 1 - np.argmax(above[::-1])
        gamma = float(abs(v_s[right] - v_s[left]) / 2.0)
    else:
        gamma = float((v_s[-1] - v_s[0]) * 0.1)
    gamma = max(gamma, 1e-10)
    return A, gamma, offset


def _extract_peak_positions(
    v_s: np.ndarray,
    sig: np.ndarray,
    A: float,
    gamma: float,
    offset: float,
) -> np.ndarray:
    """Fit peak center per row with A, gamma, offset held fixed."""
    N = sig.shape[0]
    peaks = np.empty(N)
    vs_min, vs_max = v_s[0], v_s[-1]

    for i in range(N):
        row = sig[i]

        def neg_loglik(center: float, _row=row) -> float:
            model = A / (1.0 + ((v_s - center) / gamma) ** 2) + offset
            return float(np.sum((_row - model) ** 2))

        result = minimize_scalar(neg_loglik, bounds=(vs_min, vs_max), method="bounded")
        peaks[i] = result.x

    valid = np.isfinite(peaks)
    if not np.all(valid) and np.any(valid):
        peaks[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], peaks[valid])

    return peaks


def _merge_nearby_cps(cp_indices: List[int], min_gap: int = 3) -> List[int]:
    """Merge change points that are closer than *min_gap* rows."""
    if not cp_indices:
        return []
    merged = [cp_indices[0]]
    for idx in cp_indices[1:]:
        if idx - merged[-1] >= min_gap:
            merged.append(idx)
        else:
            merged[-1] = (merged[-1] + idx) // 2
    return merged


def _fit_piecewise_linear(
    v_d: np.ndarray,
    peak_pos: np.ndarray,
    cp_indices: List[int],
) -> tuple[float, float, List[float]]:
    """Fit shared slope + per-segment intercepts via least-squares.

    Model: peak_pos[i] = alpha * v_d[i] + c_{seg(i)}

    This is a linear system with one column for alpha and one indicator
    column per segment.
    """
    N = len(v_d)
    if N < 2:
        return 0.0, float("nan"), [float(np.mean(peak_pos))]

    boundaries = [0] + sorted(cp_indices) + [N]
    # Remove zero-length segments
    boundaries = sorted(set(boundaries))
    n_seg = len(boundaries) - 1

    X = np.zeros((N, 1 + n_seg))
    X[:, 0] = v_d
    for seg_k, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        X[start:end, 1 + seg_k] = 1.0

    with np.errstate(all="ignore"):
        coeffs, _, _, _ = np.linalg.lstsq(X, peak_pos, rcond=None)

    if not np.all(np.isfinite(coeffs)):
        alpha_rough = float((peak_pos[-1] - peak_pos[0]) / (v_d[-1] - v_d[0])) if abs(v_d[-1] - v_d[0]) > 1e-15 else 0.0
        return alpha_rough, float("nan"), [float(np.mean(peak_pos))]

    alpha = float(coeffs[0])
    intercepts = [float(coeffs[1 + k]) for k in range(n_seg)]

    y_pred = X @ coeffs
    resid = peak_pos - y_pred
    dof = max(N - (1 + n_seg), 1)
    mse = float(np.sum(resid**2) / dof)

    try:
        XtX = X.T @ X
        if np.linalg.cond(XtX) < 1e12:
            cov = mse * np.linalg.inv(XtX)
            alpha_std = float(np.sqrt(max(cov[0, 0], 0.0)))
        else:
            alpha_std = float("nan")
    except np.linalg.LinAlgError:
        alpha_std = float("nan")

    return alpha, alpha_std, intercepts
