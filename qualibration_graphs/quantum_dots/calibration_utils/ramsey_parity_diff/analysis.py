"""1-D Ramsey parity-difference analysis with ±δ triangulation.

This module analyses data from a Ramsey experiment where the idle time τ
between two π/2 pulses is swept at **two symmetric detunings** ±δ from
the qubit intermediate frequency.

Each trace follows a damped cosine:

.. math::

    P(\\tau) = \\mathrm{offset} + A\\,\\exp(-\\gamma\\,\\tau)\\,
              \\cos(2\\pi f\\,\\tau + \\varphi)

The Ramsey oscillation frequency *f* equals the true detuning from
resonance.  By fitting both traces independently:

* **+δ trace** oscillates at  f₊ = |Δ − δ|
* **−δ trace** oscillates at  f₋ = |Δ + δ|

where Δ is the (signed) residual detuning of the qubit from the IF.
For |Δ| < δ this gives **Δ = (f₋ − f₊) / 2**, resolving the sign
ambiguity inherent in a single-detuning measurement.

The fit uses :func:`scipy.optimize.differential_evolution` as a global
optimiser to avoid local minima in the oscillatory landscape.

Extracted quantities
--------------------
* **freq_offset** — triangulated residual detuning Δ (Hz).
* **T2*** — dephasing time from the exponential envelope (ns),
  averaged over both traces.
* **decay_rate** — γ = 1/T₂* (1/ns).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import differential_evolution

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)


# ── Damped-cosine model ──────────────────────────────────────────────────────


def _damped_cosine(
    t: np.ndarray,
    offset: float,
    amp: float,
    freq: float,
    gamma: float,
    phi: float,
) -> np.ndarray:
    r"""Evaluate ``offset + amp * exp(-γt) * cos(2πft + φ)``.

    Parameters
    ----------
    t : array
        Time values (ns), shifted so t[0] = 0.
    offset : float
        Baseline (off-resonance parity level).
    amp : float
        Oscillation amplitude (contrast).
    freq : float
        Oscillation frequency in cycles/ns.
    gamma : float
        Exponential decay rate in 1/ns (T₂* = 1/γ).
    phi : float
        Phase offset in radians.
    """
    return offset + amp * np.exp(-gamma * t) * np.cos(
        2.0 * np.pi * freq * t + phi
    )


# ── Single-trace fit ─────────────────────────────────────────────────────────


def _fit_single_trace(
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
    detuning_hz: float,
) -> Dict[str, Any]:
    """Fit a damped cosine to one Ramsey time trace.

    Uses :func:`scipy.optimize.differential_evolution` to globally
    optimise the 5-parameter model, avoiding local minima in the
    oscillatory landscape.

    Parameters
    ----------
    pdiff : 1-D array (n_tau,)
        Parity-difference values.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.
    detuning_hz : float
        Applied drive-frequency detuning in Hz (used only for
        frequency-bound seeding).

    Returns
    -------
    dict
        ``ramsey_freq`` (Hz), ``decay_rate`` (1/ns), ``t2_star`` (ns),
        ``success`` (bool), and diagnostic arrays for plotting.
    """
    y = np.asarray(pdiff, dtype=float)
    t = np.asarray(tau_ns, dtype=float) - tau_ns[0]
    t_span = float(t[-1]) if len(t) > 1 else 1.0

    offset0 = float(np.mean(y))
    amp0 = float(np.ptp(y)) / 2.0

    # Frequency seed from nominal detuning
    freq_seed = abs(detuning_hz) * 1e-9

    # Refine from FFT
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    fft_mag = np.abs(np.fft.rfft(y - offset0))
    fft_freqs = np.fft.rfftfreq(n, dt)
    fft_mag[0] = 0.0
    fft_peak_idx = int(np.argmax(fft_mag))
    freq_fft = float(fft_freqs[fft_peak_idx]) if fft_peak_idx > 0 else freq_seed
    freq_est = freq_fft if freq_fft > 1e-7 else max(freq_seed, 1e-7)

    freq_lo = max(1e-7, freq_est * 0.2)
    freq_hi = max(freq_est * 5.0, 0.05)

    de_bounds = [
        (float(np.min(y)) - amp0, float(np.max(y)) + amp0),
        (-amp0 * 3.0, amp0 * 3.0),
        (freq_lo, freq_hi),
        (0.0, 10.0 / t_span),
        (-np.pi, np.pi),
    ]

    result: Dict[str, Any] = {
        "ramsey_freq": abs(detuning_hz),
        "decay_rate": np.nan,
        "t2_star": np.nan,
        "success": False,
        "fitted_curve": None,
        "tau_shifted": t,
        "pdiff": y,
    }

    try:
        def _objective(params):
            return np.sum((_damped_cosine(t, *params) - y) ** 2)

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
        freq_fit_hz = float(popt[2]) * 1e9
        gamma_fit = float(popt[3])
        t2 = 1.0 / gamma_fit if gamma_fit > 1e-12 else np.nan

        result["ramsey_freq"] = freq_fit_hz
        result["decay_rate"] = gamma_fit
        result["t2_star"] = t2
        result["fitted_curve"] = _damped_cosine(t, *popt)
        result["success"] = np.isfinite(t2) and t2 > 0
    except Exception:
        _logger.debug("Single-trace fit failed for detuning %.3f MHz", detuning_hz * 1e-6)

    return result


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a ±δ Ramsey parity-difference measurement.

    Attributes
    ----------
    freq_offset : float
        Triangulated residual detuning Δ = (f₋ − f₊)/2 in Hz —
        the correction to apply to the qubit drive frequency.
    t2_star : float
        Dephasing time in ns, averaged over both traces.
    decay_rate : float
        Exponential decay rate γ in 1/ns, averaged over both traces.
    freq_plus : float
        Ramsey oscillation frequency from the +δ trace (Hz).
    freq_minus : float
        Ramsey oscillation frequency from the −δ trace (Hz).
    success : bool
        ``True`` if both fits converged.
    """

    freq_offset: float
    t2_star: float
    decay_rate: float
    freq_plus: float
    freq_minus: float
    success: bool


# ── Single-qubit dual-trace analysis ─────────────────────────────────────────


def _analyse_single_qubit(
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
    detuning_hz: np.ndarray,
) -> Dict[str, Any]:
    """Analyse one qubit's ±δ Ramsey data and triangulate the frequency.

    Fits a damped cosine to each of the two detuning traces and
    computes the residual frequency offset from the difference in
    fitted oscillation frequencies.

    Parameters
    ----------
    pdiff : 2-D array (2, n_tau)
        Parity-difference data for [+δ, −δ] detunings.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.
    detuning_hz : 1-D array (2,)
        Applied detuning values [+δ, −δ] in Hz.

    Returns
    -------
    dict
        ``freq_offset`` (Hz), ``t2_star`` (ns), ``decay_rate`` (1/ns),
        ``freq_plus`` (Hz), ``freq_minus`` (Hz), ``success`` (bool),
        and ``_diag`` with per-trace diagnostic data.
    """
    # +δ trace (first row) and −δ trace (second row)
    fit_plus = _fit_single_trace(pdiff[0, :], tau_ns, float(detuning_hz[0]))
    fit_minus = _fit_single_trace(pdiff[1, :], tau_ns, float(detuning_hz[1]))

    f_plus = fit_plus["ramsey_freq"]
    f_minus = fit_minus["ramsey_freq"]

    # Triangulation: Δ = (f₋ − f₊) / 2
    freq_offset = (f_minus - f_plus) / 2.0

    # Average T2* and decay rate from both traces
    gammas = []
    t2s = []
    for fit in (fit_plus, fit_minus):
        if fit["success"] and np.isfinite(fit["decay_rate"]):
            gammas.append(fit["decay_rate"])
        if fit["success"] and np.isfinite(fit["t2_star"]):
            t2s.append(fit["t2_star"])

    decay_rate = float(np.mean(gammas)) if gammas else np.nan
    t2_star = float(np.mean(t2s)) if t2s else np.nan
    success = fit_plus["success"] and fit_minus["success"]

    _logger.debug(
        "Ramsey ±δ triangulation: f₊=%.3f MHz, f₋=%.3f MHz, "
        "Δ=%.3f MHz, T2*=%.1f ns",
        f_plus * 1e-6,
        f_minus * 1e-6,
        freq_offset * 1e-6,
        t2_star if np.isfinite(t2_star) else -1,
    )

    return {
        "freq_offset": freq_offset,
        "t2_star": float(t2_star),
        "decay_rate": float(decay_rate),
        "freq_plus": f_plus,
        "freq_minus": f_minus,
        "success": success,
        "_diag": {
            "detuning_hz": detuning_hz,
            "fit_plus": fit_plus,
            "fit_minus": fit_minus,
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit Ramsey frequency and T₂* for each qubit using ±δ triangulation.

    Expects a 2-D dataset with coordinates ``detuning`` (2 values:
    [+δ, −δ] in Hz) and ``tau`` (idle time in ns), with data variables
    ``pdiff_<qubit>`` of shape (2, n_tau).

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data.
    node : QualibrationNode
        Calibration node (provides qubit list).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name → dict of :class:`FitParameters` fields plus a
        ``_diag`` key with per-trace diagnostics for plotting.
    """
    qubits = node.namespace["qubits"]
    detuning_hz = np.asarray(ds.detuning.values, dtype=float)

    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [
            getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)
        ]

    tau_ns = np.asarray(ds.tau.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                freq_offset=0.0,
                t2_star=np.nan,
                decay_rate=np.nan,
                freq_plus=np.nan,
                freq_minus=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        result = _analyse_single_qubit(pdiff, tau_ns, detuning_hz)

        fp = FitParameters(
            freq_offset=result["freq_offset"],
            t2_star=result["t2_star"],
            decay_rate=result["decay_rate"],
            freq_plus=result["freq_plus"],
            freq_minus=result["freq_minus"],
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
        f_plus = r.get("freq_plus", 0) * 1e-6
        f_minus = r.get("freq_minus", 0) * 1e-6
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"f₊={f_plus:.3f} MHz, f₋={f_minus:.3f} MHz, "
            f"Δ={freq_off:.3f} MHz, "
            f"T2*={t2:.0f} ns, "
            f"gamma={gamma:.5f} 1/ns, "
            f"success={success}"
        )
        log_callable(msg)
