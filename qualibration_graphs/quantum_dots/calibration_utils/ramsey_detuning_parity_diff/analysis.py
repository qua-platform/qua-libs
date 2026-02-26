"""Two-τ Ramsey detuning-sweep analysis — independent fits + joint extraction.

This module analyses data from a Ramsey experiment where the
**drive-frequency detuning δ** is swept at **two fixed idle times**
(τ_short and τ_long).

Stage 1 — Independent per-trace cosine fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each trace follows a cosine whose frequency *f* is **not assumed known**
(the effective evolution time includes phase accrual during the finite-
duration π/2 pulses):

.. math::

    P_i(\\delta) = \\mathrm{bg}_i
        + C_i\\,\\cos(2\\pi f_i\\,\\delta)
        + S_i\\,\\sin(2\\pi f_i\\,\\delta)

A **profiled differential-evolution** optimises over the single
non-linear parameter *f*, while the linear parameters (bg, C, S) are
solved exactly via ``np.linalg.lstsq`` at each candidate *f*.  This
makes the per-trace fit a fast 1-D global search.

From the solution:

* amplitude  A_i = √(C_i² + S_i²)
* resonance  δ₀_i = +atan2(S_i, C_i) / (2π f_i)

Stage 2 — Joint extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resonance detuning δ₀ is taken as the amplitude-weighted mean of
the per-trace estimates δ₀_short and δ₀_long.

The decay rate exploits the fact that the pulse contribution to the
amplitude cancels in the ratio:

.. math::

    \\gamma = -\\frac{\\ln(A_\\mathrm{long} / A_\\mathrm{short})}
                     {\\tau_\\mathrm{long} - \\tau_\\mathrm{short}}

Extracted quantities
--------------------
* **freq_offset** — resonance detuning δ₀ (Hz).
* **contrast** — oscillation amplitude from the short-τ trace.
* **decay_rate** — exponential decay rate γ (1/ns).
* **t2_star** — dephasing time 1/γ (ns).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import differential_evolution

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a two-τ detuning-swept Ramsey measurement.

    Attributes
    ----------
    freq_offset : float
        Resonance detuning δ₀ (Hz) — the correction to subtract from
        the qubit intermediate frequency so the drive sits on resonance.
    contrast : float
        Oscillation amplitude A from the short-τ trace.
    decay_rate : float
        Exponential decay rate γ (1/ns), from amplitude ratio.
    t2_star : float
        Dephasing time T₂* = 1/γ (ns).
    success : bool
        ``True`` if both per-trace fits converged.
    """

    freq_offset: float
    contrast: float
    decay_rate: float
    t2_star: float
    success: bool


# ── Per-trace profiled DE fit ────────────────────────────────────────────────


def _fit_single_trace(
    pdiff: np.ndarray,
    detuning_hz: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit a single detuning-sweep trace with free oscillation frequency.

    Uses a profiled DE approach: for each candidate frequency *f* the
    linear parameters (bg, C, S) are solved exactly, so DE searches only
    the 1-D non-linear parameter *f*.

    Parameters
    ----------
    pdiff : 1-D array (n_det,)
        Parity-difference values.
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.

    Returns
    -------
    dict
        ``osc_freq`` (Hz⁻¹), ``amplitude``, ``delta0`` (Hz),
        ``bg``, ``C``, ``S``, ``fitted_curve``, ``success``.
    """
    y = np.asarray(pdiff, dtype=float)
    delta = np.asarray(detuning_hz, dtype=float)
    n = len(delta)
    d_step = abs(delta[1] - delta[0]) if n > 1 else 1.0
    d_span = abs(delta[-1] - delta[0]) if n > 1 else 1.0

    result: Dict[str, Any] = {
        "osc_freq": np.nan,
        "amplitude": 0.0,
        "delta0": 0.0,
        "bg": np.nan,
        "C": np.nan,
        "S": np.nan,
        "fitted_curve": np.full_like(y, np.nan),
        "success": False,
    }

    # ── FFT seed for oscillation frequency ───────────────────────────
    y_centered = y - np.mean(y)
    fft_mag = np.abs(np.fft.rfft(y_centered))
    fft_freqs = np.fft.rfftfreq(n, d_step)
    fft_mag[0] = 0.0
    peak_idx = int(np.argmax(fft_mag))
    f_fft = float(fft_freqs[peak_idx]) if peak_idx > 0 else 1.0 / d_span

    # DE bounds for f: from ~1 period across span to Nyquist
    f_lo = max(0.5 / d_span, 1e-9)
    f_hi = 0.5 / d_step
    # Ensure FFT seed is within bounds
    f_fft = np.clip(f_fft, f_lo, f_hi)

    def _profile_residual(f_arr):
        f = f_arr[0]
        phase = 2.0 * np.pi * f * delta
        A_mat = np.column_stack(
            [
                np.ones(n),
                np.cos(phase),
                np.sin(phase),
            ]
        )
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        return float(np.sum((y - A_mat @ coeffs) ** 2))

    try:
        de_result = differential_evolution(
            _profile_residual,
            bounds=[(f_lo, f_hi)],
            seed=42,
            maxiter=1000,
            tol=1e-12,
            polish=True,
            popsize=30,
        )
        f_best = float(de_result.x[0])

        # Solve linear system at best f
        phase = 2.0 * np.pi * f_best * delta
        A_mat = np.column_stack(
            [
                np.ones(n),
                np.cos(phase),
                np.sin(phase),
            ]
        )
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        bg, C, S = coeffs

        amplitude = float(np.hypot(C, S))
        # P(δ) = bg + A·cos(2πf(δ − δ₀))  ⟹  δ₀ = +atan2(S,C)/(2πf)
        delta0 = float(np.arctan2(S, C) / (2.0 * np.pi * f_best))

        result["osc_freq"] = f_best
        result["amplitude"] = amplitude
        result["delta0"] = delta0
        result["bg"] = float(bg)
        result["C"] = float(C)
        result["S"] = float(S)
        result["fitted_curve"] = A_mat @ coeffs
        result["success"] = np.isfinite(delta0) and amplitude > 1e-6

        _logger.debug(
            "Per-trace fit: f=%.4e Hz⁻¹, A=%.4f, δ₀=%.3f MHz",
            f_best,
            amplitude,
            delta0 * 1e-6,
        )
    except Exception:
        _logger.debug("Per-trace profiled DE fit failed", exc_info=True)

    return result


# ── Single-qubit two-τ analysis ──────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff_2d: np.ndarray,
    detuning_hz: np.ndarray,
    tau_ns: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit a single qubit's two-τ Ramsey data (independent + joint).

    Stage 1: fits each trace independently with free oscillation
    frequency via profiled DE.  Stage 2: extracts shared resonance
    δ₀ (amplitude-weighted mean) and decay rate γ from the amplitude
    ratio.

    Parameters
    ----------
    pdiff_2d : 2-D array (n_tau, n_det)
        Parity-difference data for each idle time.
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds (short, long).

    Returns
    -------
    dict
        ``freq_offset`` (Hz), ``contrast``, ``decay_rate`` (1/ns),
        ``t2_star`` (ns), ``success`` (bool), and ``_traces`` with
        per-trace fit diagnostics.
    """
    y = np.asarray(pdiff_2d, dtype=float)
    delta = np.asarray(detuning_hz, dtype=float)
    taus = np.asarray(tau_ns, dtype=float)
    n_tau = len(taus)

    # ── Stage 1: Independent per-trace fits ──────────────────────────
    trace_fits: List[Dict[str, Any]] = []
    for i in range(n_tau):
        fit = _fit_single_trace(y[i], delta)
        fit["tau_ns"] = float(taus[i])
        trace_fits.append(fit)

    all_success = all(tf["success"] for tf in trace_fits)

    result: Dict[str, Any] = {
        "freq_offset": 0.0,
        "contrast": 0.0,
        "decay_rate": np.nan,
        "t2_star": np.nan,
        "success": False,
        "fitted_curves": None,
        "_traces": trace_fits,
    }

    if not all_success:
        _logger.debug("Not all per-trace fits succeeded")
        # Collect fitted curves even from partial success
        result["fitted_curves"] = np.array([tf["fitted_curve"] for tf in trace_fits])
        return result

    # ── Stage 2: Joint extraction ────────────────────────────────────
    amplitudes = np.array([tf["amplitude"] for tf in trace_fits])
    delta0s = np.array([tf["delta0"] for tf in trace_fits])

    # Resonance: amplitude-weighted mean of per-trace δ₀
    weights = amplitudes / np.sum(amplitudes)
    freq_offset = float(np.sum(weights * delta0s))

    # Contrast from the short-τ trace (index 0)
    contrast = float(amplitudes[0])

    # Decay rate from amplitude ratio (pulse contributions cancel)
    tau_diff = float(taus[-1] - taus[0])
    if tau_diff > 0 and amplitudes[0] > 1e-12 and amplitudes[-1] > 1e-12:
        ratio = amplitudes[-1] / amplitudes[0]
        if 0 < ratio < 1:
            gamma = -np.log(ratio) / tau_diff
        elif ratio >= 1:
            gamma = 0.0
        else:
            gamma = np.nan
    else:
        gamma = np.nan

    t2 = 1.0 / gamma if np.isfinite(gamma) and gamma > 1e-12 else np.nan

    result["freq_offset"] = freq_offset
    result["contrast"] = contrast
    result["decay_rate"] = float(gamma) if np.isfinite(gamma) else np.nan
    result["t2_star"] = float(t2) if np.isfinite(t2) else np.nan
    result["fitted_curves"] = np.array([tf["fitted_curve"] for tf in trace_fits])
    result["success"] = np.isfinite(freq_offset) and contrast > 1e-6

    _logger.debug(
        "Joint extraction: δ₀=%.3f MHz (short=%.3f, long=%.3f), " "A_short=%.4f, A_long=%.4f, γ=%.5f 1/ns, T2*=%.1f ns",
        freq_offset * 1e-6,
        delta0s[0] * 1e-6,
        delta0s[-1] * 1e-6,
        amplitudes[0],
        amplitudes[-1],
        gamma if np.isfinite(gamma) else -1,
        t2 if np.isfinite(t2) else -1,
    )

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit resonance detuning via independent per-trace DE + joint extraction.

    Expects a 2-D dataset with coordinates ``tau`` (ns) and
    ``detuning`` (Hz), with data variables ``pdiff_<qubit>`` of
    shape ``(n_tau, n_det)``.

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
        qubit name → dict of :class:`FitParameters` fields plus
        ``_diag`` with raw solver outputs for plotting.
    """
    qubits = node.namespace["qubits"]
    detuning_hz = np.asarray(ds.detuning.values, dtype=float)
    tau_ns = np.asarray(ds.tau.values, dtype=float)

    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                freq_offset=0.0,
                contrast=0.0,
                decay_rate=np.nan,
                t2_star=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        raw = _analyse_single_qubit(pdiff, detuning_hz, tau_ns)

        fp = FitParameters(
            freq_offset=raw["freq_offset"],
            contrast=raw["contrast"],
            decay_rate=raw["decay_rate"],
            t2_star=raw["t2_star"],
            success=raw["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_results[qname]["_diag"] = raw

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted Ramsey detuning parameters for all qubits.

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
        freq_off_mhz = r.get("freq_offset", 0) * 1e-6
        contrast = r.get("contrast", 0)
        gamma = r.get("decay_rate", np.nan)
        t2 = r.get("t2_star", np.nan)
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"δ₀={freq_off_mhz:.4f} MHz, "
            f"A_short={contrast:.4f}, "
            f"γ={gamma:.5f} 1/ns, "
            f"T2*={t2:.0f} ns, "
            f"success={success}"
        )
        log_callable(msg)
