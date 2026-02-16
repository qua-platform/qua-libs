"""1-D Ramsey parity-difference analysis: damped-cosine fit via DE.

This module analyses data from a single-detuning Ramsey experiment where
the idle time τ between two π/2 pulses is swept at a fixed drive-frequency
offset δ.  The parity-difference signal is expected to follow:

.. math::

    P(\\tau) = \\mathrm{offset} + A\\,\\exp(-\\gamma\\,\\tau)\\,
              \\cos(2\\pi f\\,\\tau + \\varphi)

where *f* ≈ |δ| (converted to cycles/ns) is the Ramsey oscillation
frequency, *γ* = 1/T₂* is the dephasing rate, and *φ* captures the
initial phase.

The fit uses :func:`scipy.optimize.differential_evolution` as a global
optimiser to avoid local minima in the oscillatory landscape, followed
by a local polish step.

Extracted quantities
--------------------
* **freq_offset** — residual frequency offset (Hz) between the applied
  detuning and the true qubit splitting.
* **T2*** — dephasing time from the exponential envelope (ns).
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
        Oscillation frequency in cycles/ns (= |detuning_Hz| × 1e-9).
    gamma : float
        Exponential decay rate in 1/ns (T₂* = 1/γ).
    phi : float
        Phase offset in radians.
    """
    return offset + amp * np.exp(-gamma * t) * np.cos(
        2.0 * np.pi * freq * t + phi
    )


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a 1-D Ramsey parity-difference measurement.

    Attributes
    ----------
    freq_offset : float
        Residual frequency offset in Hz — the difference between the
        fitted oscillation frequency and the nominal detuning.
    t2_star : float
        Dephasing time in ns (1/γ from the exponential envelope).
    decay_rate : float
        Exponential decay rate γ in 1/ns.
    ramsey_freq : float
        Fitted Ramsey oscillation frequency in Hz.
    success : bool
        ``True`` if the fit converged and T₂* is finite and positive.
    """

    freq_offset: float
    t2_star: float
    decay_rate: float
    ramsey_freq: float
    success: bool


# ── Single-qubit 1-D analysis ───────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
    detuning_hz: float,
) -> Dict[str, Any]:
    """Analyse one qubit's 1-D Ramsey time trace.

    Fits the damped cosine model to the parity-difference vs idle time
    using :func:`scipy.optimize.differential_evolution`.  The global
    optimiser explores the full parameter space, which is crucial because
    the oscillatory landscape has many local minima for gradient-based
    methods.

    Parameters
    ----------
    pdiff : 1-D array (n_tau,)
        Parity-difference values.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.
    detuning_hz : float
        Nominal drive-frequency detuning in Hz.

    Returns
    -------
    dict
        Keys: ``freq_offset`` (Hz), ``t2_star`` (ns), ``decay_rate``
        (1/ns), ``ramsey_freq`` (Hz), ``success`` (bool), and ``_diag``
        containing diagnostic arrays for plotting.
    """
    y = np.asarray(pdiff, dtype=float)
    t = np.asarray(tau_ns, dtype=float) - tau_ns[0]
    t_span = float(t[-1]) if len(t) > 1 else 1.0

    # ── Initial estimates from the data ───────────────────────────────────
    offset0 = float(np.mean(y))
    amp0 = float(np.ptp(y)) / 2.0

    # Frequency seed: nominal detuning converted to cycles/ns
    freq_seed = abs(detuning_hz) * 1e-9

    # Refine frequency from FFT
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    fft_mag = np.abs(np.fft.rfft(y - offset0))
    fft_freqs = np.fft.rfftfreq(n, dt)
    # Exclude DC bin
    fft_mag[0] = 0.0
    fft_peak_idx = int(np.argmax(fft_mag))
    freq_fft = float(fft_freqs[fft_peak_idx]) if fft_peak_idx > 0 else freq_seed

    # Use FFT estimate if it's reasonable, otherwise fall back to seed
    freq_est = freq_fft if freq_fft > 1e-7 else max(freq_seed, 1e-7)

    # ── Differential evolution bounds ─────────────────────────────────────
    # (offset, amp, freq, gamma, phi)
    freq_lo = max(1e-7, freq_est * 0.2)
    freq_hi = max(freq_est * 5.0, 0.05)

    de_bounds = [
        (float(np.min(y)) - amp0, float(np.max(y)) + amp0),  # offset
        (-amp0 * 3.0, amp0 * 3.0),                            # amp (signed)
        (freq_lo, freq_hi),                                    # freq (cyc/ns)
        (0.0, 10.0 / t_span),                                 # gamma (1/ns)
        (-np.pi, np.pi),                                       # phi
    ]

    fitted_curve = None
    freq_offset = 0.0
    decay_rate = np.nan
    t2_star = np.nan
    ramsey_freq = abs(detuning_hz)
    success = False

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
        offset_fit, amp_fit, freq_fit, gamma_fit, phi_fit = popt

        # Convert fitted frequency back to Hz
        ramsey_freq = float(freq_fit) * 1e9  # cycles/ns -> Hz
        freq_offset = ramsey_freq - abs(detuning_hz)
        decay_rate = float(gamma_fit)
        t2_star = 1.0 / decay_rate if decay_rate > 1e-12 else np.nan
        fitted_curve = _damped_cosine(t, *popt)
        success = np.isfinite(t2_star) and t2_star > 0

        _logger.debug(
            "Ramsey 1-D fit (DE): f_ramsey=%.3f MHz, "
            "f_offset=%.3f MHz, gamma=%.5f 1/ns, T2*=%.1f ns",
            ramsey_freq * 1e-6,
            freq_offset * 1e-6,
            decay_rate,
            t2_star if np.isfinite(t2_star) else -1,
        )
    except Exception:
        _logger.debug("Ramsey 1-D fit failed; returning NaN estimates")

    return {
        "freq_offset": freq_offset,
        "t2_star": float(t2_star),
        "decay_rate": float(decay_rate),
        "ramsey_freq": float(ramsey_freq),
        "success": success,
        "_diag": {
            "tau_shifted": t,
            "pdiff": y,
            "fitted_curve": fitted_curve,
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit Ramsey oscillation frequency and T₂* for each qubit.

    Iterates over every ``pdiff_<qubit>`` variable in the dataset and
    fits a damped cosine to each.

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data with coordinate ``tau`` (ns) and data
        variables ``pdiff_<qubit>``.
    node : QualibrationNode
        Calibration node (provides qubit list and ``frequency_detuning_in_mhz``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name -> dict of :class:`FitParameters` fields plus a
        ``_diag`` key with diagnostic arrays for plotting.
    """
    qubits = node.namespace["qubits"]
    detuning_hz = float(node.parameters.frequency_detuning_in_mhz) * 1e6

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
                ramsey_freq=abs(detuning_hz),
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
            ramsey_freq=result["ramsey_freq"],
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
        Qubit name -> fit-result dict (as returned by :func:`fit_raw_data`).
    log_callable : callable, optional
        Logging function; defaults to ``_logger.info``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        freq_off = r.get("freq_offset", 0) * 1e-6
        t2 = r.get("t2_star", 0)
        gamma = r.get("decay_rate", 0)
        ramsey_f = r.get("ramsey_freq", 0) * 1e-6
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"f_ramsey={ramsey_f:.3f} MHz, "
            f"freq_offset={freq_off:.3f} MHz, "
            f"T2*={t2:.0f} ns, "
            f"gamma={gamma:.5f} 1/ns, "
            f"success={success}"
        )
        log_callable(msg)
