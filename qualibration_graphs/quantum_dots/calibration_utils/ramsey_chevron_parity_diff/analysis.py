"""Ramsey chevron analysis: mean-parity resonance + T2* extraction.

Strategy
--------
1. **Resonance finding** — The tau-averaged parity vs detuning is fitted
   to the exact finite-window model using ``differential_evolution``
   (global optimizer).  The model uses the physical (unshifted) tau
   values and a combined exponential + Gaussian decay envelope:

   .. math::

       \\langle P \\rangle(\\delta) = bg + \\frac{A}{N}
           \\sum_i \\exp\\!\\bigl(-\\gamma\\,\\tau_i - (\\sigma_g\\,\\tau_i)^2\\bigr)\\,
           \\cos\\!\\bigl(2\\pi\\,(\\delta - \\delta_0)\\,\\tau_i \\cdot 10^{-9}\\bigr)

   The amplitude can be positive (peak) or negative (dip), so no
   prior knowledge of the resonance sign is needed.

2. **T2* extraction** — The effective T2* is the 1/e time of the
   combined envelope ``exp(-γτ - (σ_g τ)²)``.  The primary estimate
   comes from the mean-parity model; an exponential-decay fit to the
   near-resonance time trace provides a fallback when the decay is
   poorly constrained.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, differential_evolution

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)


# ── Mean-parity resonance model ──────────────────────────────────────────────


def _make_mean_ramsey_model(tau_ns: np.ndarray):
    r"""Return a model for the tau-averaged Ramsey parity vs detuning.

    The exact mean over the finite tau grid is

    .. math::

        \langle P \rangle(\delta) = bg + \frac{A}{N}
            \sum_i \exp\!\bigl(-\gamma\,\tau_i - (\sigma_g\,\tau_i)^2\bigr)\,
            \cos\!\bigl(2\pi\,(\delta - \delta_0)\,\tau_i \cdot 10^{-9}\bigr)

    Uses the raw (unshifted) τ values so that the cosine phase and
    exponential envelope are physically exact.

    Parameters (of the returned callable)
    --------------------------------------
    x : array — detuning in Hz
    amp : float — peak contrast (positive = peak, negative = dip)
    x0 : float — resonance position (Hz)
    gamma : float — exponential decay rate in 1/ns
    sigma_g : float — Gaussian decay rate in 1/ns  (T₂_gauss = 1/σ_g)
    bg : float — off-resonance baseline
    """
    tau = np.asarray(tau_ns, dtype=float)

    def _model(x, amp, x0, gamma, sigma_g, bg):
        delta_hz = np.atleast_1d(np.asarray(x, dtype=float)) - x0
        # phase: outer product (n_det, n_tau)
        phase = 2.0 * np.pi * delta_hz[:, None] * tau[None, :] * 1e-9
        envelope = np.exp(-gamma * tau - (sigma_g * tau) ** 2)[None, :]
        result = bg + amp * np.mean(envelope * np.cos(phase), axis=1)
        return result if np.ndim(x) > 0 else float(result[0])

    return _model


# ── Effective T2* from combined envelope ──────────────────────────────────────


def _effective_t2_star(gamma: float, sigma_g: float) -> float:
    r"""Compute the 1/e time of exp(-γτ - (σ_g τ)²).

    Solves  σ_g²τ² + γτ − 1 = 0  for the positive root.

    Special cases:
    - σ_g = 0 → T₂* = 1/γ   (pure exponential)
    - γ = 0  → T₂* = 1/σ_g  (pure Gaussian)
    """
    if sigma_g < 1e-12:
        return 1.0 / gamma if gamma > 1e-12 else np.nan
    discriminant = gamma**2 + 4.0 * sigma_g**2
    return (-gamma + np.sqrt(discriminant)) / (2.0 * sigma_g**2)


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a Ramsey chevron measurement.

    Attributes
    ----------
    freq_offset : float
        Resonance frequency offset in Hz — the correction to apply to
        the qubit drive frequency.
    t2_star : float
        Effective dephasing time in ns, defined as the 1/e time of the
        combined envelope ``exp(-γτ - (σ_g τ)²)``.
    decay_rate : float
        Exponential decay component γ in 1/ns.
    gauss_decay_rate : float
        Gaussian decay component σ_g in 1/ns (T₂_gauss = 1/σ_g).
    success : bool
        ``True`` if the fit converged and T2* is finite and positive.
    """

    freq_offset: float
    t2_star: float
    decay_rate: float
    gauss_decay_rate: float
    success: bool


# ── T2* validation and fallback ──────────────────────────────────────────────


def _fit_exponential_decay(
    tau_ns: np.ndarray,
    trace: np.ndarray,
    t_span: float,
) -> float | None:
    """Fit a simple exponential decay ``bg + A·exp(-γt)`` to a time trace.

    Used as a fallback T2* estimator when the mean-parity model cannot
    reliably constrain the decay rate (e.g. when T2* exceeds the
    measurement window).

    Parameters
    ----------
    tau_ns : array
        Idle-time values in nanoseconds.
    trace : array
        Averaged parity-difference values near resonance.
    t_span : float
        Total duration of the measurement window (ns).

    Returns
    -------
    float or None
        Exponential decay rate γ (1/ns), or ``None`` if the fit is
        unreliable or the signal is too weak.
    """
    t = tau_ns - tau_ns[0]
    y = np.asarray(trace, dtype=float)

    bg0 = float(np.mean(y[-len(y) // 4 :]))
    amp0 = float(y[0]) - bg0
    if abs(amp0) < 0.01:
        return None
    gamma0 = 2.0 / t_span

    def _model(t, bg, amp, gamma):
        return bg + amp * np.exp(-gamma * t)

    try:
        popt, pcov = curve_fit(
            _model,
            t,
            y,
            p0=[bg0, amp0, gamma0],
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0 / t_span]),
            maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))
        gamma = float(popt[2])
        if gamma < 2e-6 or (perr[2] > gamma * 5):
            return None
        return gamma
    except Exception:
        return None


def _validate_t2_star(
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
    freq_offset: float,
    detuning_hz: np.ndarray,
    resonance_idx: int,
    decay_rate_de: float,
    sigma_g_de: float,
    t2_star_de: float,
) -> tuple[float, float, float]:
    """Validate T2* from the DE fit and fall back if poorly constrained.

    If T2* from the mean-parity model is much larger than the
    measurement window, the data cannot distinguish any decay from
    zero.  In that case, try an exponential fit to the near-resonance
    averaged time trace.

    Parameters
    ----------
    pdiff : 2-D array (n_det, n_tau)
        Parity-difference matrix.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.
    freq_offset : float
        Resonance frequency offset from the DE fit (Hz).
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.
    resonance_idx : int
        Index into *detuning_hz* closest to resonance.
    decay_rate_de, sigma_g_de, t2_star_de : float
        Values from the mean-parity DE fit.

    Returns
    -------
    (gamma, sigma_g, t2_star) : tuple of float
        Best available decay-rate components and effective T2* (ns).
    """
    n_det, n_tau = pdiff.shape
    t_span = float(tau_ns[-1] - tau_ns[0]) if n_tau > 1 else 1.0

    # Accept the DE result if T2* is physically constrained by the data.
    if np.isfinite(t2_star_de) and 0 < t2_star_de < 5.0 * t_span:
        return decay_rate_de, sigma_g_de, t2_star_de

    # Fallback: exponential decay of near-resonance averaged time trace
    half_w = max(1, n_det // 20)
    lo = max(0, resonance_idx - half_w)
    hi = min(n_det, resonance_idx + half_w + 1)
    near_trace = np.mean(pdiff[lo:hi, :], axis=0)
    gamma_exp = _fit_exponential_decay(tau_ns, near_trace, t_span)
    if gamma_exp is not None:
        return gamma_exp, 0.0, 1.0 / gamma_exp

    return decay_rate_de, sigma_g_de, t2_star_de


# ── Single-qubit 2D analysis ────────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff: np.ndarray,
    detuning_hz: np.ndarray,
    tau_ns: np.ndarray,
) -> Dict[str, Any]:
    """Analyse one qubit's 2-D Ramsey chevron and extract key parameters.

    The analysis proceeds in two stages:

    1. **Resonance finding** — Fit the tau-averaged parity profile to the
       exact finite-window sum-of-cosines model using
       ``differential_evolution``.  The model includes a combined
       exponential + Gaussian decay envelope (5 parameters: amp, δ₀, γ,
       σ_g, bg).  This global optimizer avoids local minima in the
       multi-modal landscape.
    2. **T2* validation** — Accept the DE fit's T2* if it is well
       constrained (< 5× the measurement window), otherwise fall back to
       an exponential-decay fit of the near-resonance time trace.

    Parameters
    ----------
    pdiff : 2-D array (n_det, n_tau)
        Parity-difference measurement matrix.
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.

    Returns
    -------
    dict
        Keys: ``freq_offset`` (Hz), ``t2_star`` (ns), ``decay_rate``
        (1/ns), ``gauss_decay_rate`` (1/ns), ``success`` (bool), and
        ``_diag`` (diagnostic data for plotting).
    """
    n_det, n_tau = pdiff.shape

    # ── Step 1: Fit mean parity vs detuning ──────────────────────────────
    mean_parity = np.mean(pdiff, axis=1)
    model = _make_mean_ramsey_model(tau_ns)

    # Find the most prominent feature (works for both peak and dip)
    median_val = float(np.median(mean_parity))
    abs_dev = np.abs(mean_parity - median_val)
    extremum_idx = int(np.argmax(abs_dev))

    freq_offset = float(detuning_hz[extremum_idx])
    resonance_idx = extremum_idx
    mean_parity_fit = None
    decay_rate = np.nan
    sigma_g = 0.0
    t2_star = np.nan

    try:
        ptp = float(np.ptp(mean_parity))
        det_min, det_max = float(detuning_hz.min()), float(detuning_hz.max())
        t_span = float(tau_ns[-1] - tau_ns[0]) if n_tau > 1 else 1.0
        y_min, y_max = float(mean_parity.min()), float(mean_parity.max())

        # Parameter bounds: (amp, x0, gamma, sigma_g, bg)
        de_bounds = [
            (-ptp * 3, ptp * 3),  # amp
            (det_min, det_max),  # x0 (Hz)
            (0.0, 10.0 / t_span),  # gamma (1/ns)
            (0.0, 10.0 / t_span),  # sigma_g (1/ns)
            (y_min - ptp, y_max + ptp),  # bg
        ]

        def _objective(params):
            return np.sum((model(detuning_hz, *params) - mean_parity) ** 2)

        de_result = differential_evolution(
            _objective,
            de_bounds,
            seed=42,
            maxiter=1000,
            tol=1e-10,
            polish=True,
            popsize=20,
        )
        popt = de_result.x
        freq_offset = float(popt[1])
        decay_rate = float(popt[2])
        sigma_g = float(popt[3])
        resonance_idx = int(np.argmin(np.abs(detuning_hz - freq_offset)))
        t2_star = _effective_t2_star(decay_rate, sigma_g)
        mean_parity_fit = model(detuning_hz, *popt)
        is_peak = popt[0] > 0
        _logger.debug(
            "Ramsey mean-parity fit (DE, %s): f_offset=%.3f MHz, " "gamma=%.5f 1/ns, sigma_g=%.5f 1/ns, T2*=%.1f ns",
            "peak" if is_peak else "dip",
            freq_offset * 1e-6,
            decay_rate,
            sigma_g,
            t2_star if np.isfinite(t2_star) else -1,
        )
    except Exception:
        _logger.debug(
            "Mean-parity fit failed; using raw extremum at %.3f MHz",
            freq_offset * 1e-6,
        )

    # ── Step 2: Validate / refine T2* ────────────────────────────────────
    decay_rate, sigma_g, t2_star = _validate_t2_star(
        pdiff,
        tau_ns,
        freq_offset,
        detuning_hz,
        resonance_idx,
        decay_rate,
        sigma_g,
        t2_star,
    )

    success = np.isfinite(t2_star) and t2_star > 0

    return {
        "freq_offset": freq_offset,
        "t2_star": float(t2_star),
        "decay_rate": float(decay_rate),
        "gauss_decay_rate": float(sigma_g),
        "success": success,
        "_diag": {
            "mean_parity": mean_parity,
            "mean_parity_fit": mean_parity_fit,
            "resonance_idx": resonance_idx,
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit resonance frequency offset and T2* for each qubit.

    Iterates over every ``pdiff_<qubit>`` variable in the dataset and
    runs :func:`_analyse_single_qubit` on each.

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data with coordinates ``detuning`` (Hz) and
        ``tau`` (ns), and data variables ``pdiff_<qubit>``.
    node : QualibrationNode
        Calibration node (provides ``node.namespace["qubits"]``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name → dict of :class:`FitParameters` fields plus a
        ``_diag`` key with diagnostic arrays for plotting.
    """
    qubits = node.namespace["qubits"]
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]

    detuning_hz = np.asarray(ds.detuning.values, dtype=float)
    tau_ns = np.asarray(ds.tau.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                freq_offset=0.0,
                t2_star=np.nan,
                decay_rate=np.nan,
                gauss_decay_rate=0.0,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        result = _analyse_single_qubit(pdiff, detuning_hz, tau_ns)

        fp = FitParameters(
            freq_offset=result["freq_offset"],
            t2_star=result["t2_star"],
            decay_rate=result["decay_rate"],
            gauss_decay_rate=result["gauss_decay_rate"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_results[qname]["_diag"] = result.get("_diag")

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted Ramsey parameters for all qubits.

    Parameters
    ----------
    fit_results : dict
        Qubit name → fit-result dict (as returned by :func:`fit_raw_data`).
    log_callable : callable, optional
        Logging function; defaults to ``_logger.info``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        freq_off = r.get("freq_offset", 0) * 1e-6
        t2 = r.get("t2_star", 0)
        gamma = r.get("decay_rate", 0)
        sigma_g = r.get("gauss_decay_rate", 0)
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"freq_offset={freq_off:.3f} MHz, "
            f"T2*={t2:.0f} ns, "
            f"gamma={gamma:.5f} 1/ns, "
            f"sigma_g={sigma_g:.5f} 1/ns, "
            f"success={success}"
        )
        log_callable(msg)
