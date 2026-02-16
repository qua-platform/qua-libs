"""Two-τ Ramsey detuning-sweep analysis via joint differential evolution.

This module analyses data from a Ramsey experiment where the
**drive-frequency detuning δ** is swept at **two fixed idle times**
(τ_short and τ_long).

At each idle time τᵢ, the parity-difference signal is:

.. math::

    P_i(\\delta) = \\mathrm{bg}
        + A_0\\,\\exp(-\\gamma\\,\\tau_i)\\,
          \\cos\\bigl(2\\pi\\,\\tau_i \\cdot 10^{-9}\\,(\\delta - \\delta_0)\\bigr)

Using two idle times creates a **Vernier** effect: the short-τ trace
produces wide fringes (coarse resonance localisation) while the long-τ
trace produces narrow fringes (fine precision).  The true resonance δ₀
is the unique detuning where both traces are simultaneously in phase.
Meanwhile the amplitude ratio between traces gives the exponential
decay rate:

.. math::

    \\gamma = -\\frac{\\ln(A_\\mathrm{long}/A_\\mathrm{short})}{\\tau_\\mathrm{long} - \\tau_\\mathrm{short}}

A single :func:`scipy.optimize.differential_evolution` fit with
**shared parameters** (bg, A₀, δ₀, γ) across both traces provides a
globally optimal, over-constrained solution.

Extracted quantities
--------------------
* **freq_offset** — resonance detuning δ₀ (Hz), the correction to
  apply to the qubit intermediate frequency.
* **contrast** — bare oscillation amplitude A₀ (before decay).
* **decay_rate** — exponential decay rate γ (1/ns).
* **t2_star** — dephasing time 1/γ (ns).
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
        Bare oscillation amplitude A₀ (before exponential decay).
    decay_rate : float
        Exponential decay rate γ (1/ns).
    t2_star : float
        Dephasing time T₂* = 1/γ (ns).
    success : bool
        ``True`` if the DE fit converged with finite parameters.
    """

    freq_offset: float
    contrast: float
    decay_rate: float
    t2_star: float
    success: bool


# ── Joint two-τ model ────────────────────────────────────────────────────────


def _joint_model(
    params: tuple,
    delta_hz: np.ndarray,
    tau_ns: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the two-τ Ramsey model for all traces.

    Parameters
    ----------
    params : (bg, amp, delta0, gamma)
        bg     — baseline parity level
        amp    — bare oscillation amplitude (before decay)
        delta0 — resonance detuning (Hz)
        gamma  — exponential decay rate (1/ns)
    delta_hz : 1-D array (n_det,)
        Detuning values in Hz.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.

    Returns
    -------
    model : 2-D array (n_tau, n_det)
    """
    bg, amp, delta0, gamma = params
    delta_hz = np.asarray(delta_hz, dtype=float)
    tau_ns = np.asarray(tau_ns, dtype=float)

    model = np.empty((len(tau_ns), len(delta_hz)), dtype=float)
    for i, tau in enumerate(tau_ns):
        f = tau * 1e-9  # oscillation "frequency" in Hz⁻¹
        envelope = amp * np.exp(-gamma * tau)
        model[i, :] = bg + envelope * np.cos(
            2.0 * np.pi * f * (delta_hz - delta0)
        )
    return model


# ── Single-qubit analysis ────────────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff_2d: np.ndarray,
    detuning_hz: np.ndarray,
    tau_ns: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit a single qubit's two-τ detuning-swept Ramsey data.

    Uses differential evolution to globally optimise the 4-parameter
    joint model (bg, A₀, δ₀, γ) across both traces simultaneously.

    Parameters
    ----------
    pdiff_2d : 2-D array (n_tau, n_det)
        Parity-difference data for each idle time.
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.

    Returns
    -------
    dict
        ``freq_offset`` (Hz), ``contrast``, ``decay_rate`` (1/ns),
        ``t2_star`` (ns), ``success`` (bool), and diagnostic arrays
        for plotting.
    """
    y = np.asarray(pdiff_2d, dtype=float)
    delta = np.asarray(detuning_hz, dtype=float)
    taus = np.asarray(tau_ns, dtype=float)

    bg0 = float(np.mean(y))
    amp0 = float(np.ptp(y)) / 2.0
    det_span = float(delta[-1] - delta[0]) if len(delta) > 1 else 1e6
    tau_max = float(np.max(taus))

    # DE bounds: [bg, amp, delta0, gamma]
    de_bounds = [
        (float(np.min(y)) - 0.1, float(np.max(y)) + 0.1),  # bg
        (-1.0, 1.0),  # amp (sign handles peak vs trough)
        (float(delta[0]), float(delta[-1])),  # delta0 within sweep
        (0.0, 10.0 / tau_max),  # gamma: 0 to fast-decay limit
    ]

    result: Dict[str, Any] = {
        "freq_offset": 0.0,
        "contrast": 0.0,
        "decay_rate": np.nan,
        "t2_star": np.nan,
        "success": False,
        "fitted_curves": None,
    }

    try:

        def _objective(params):
            model = _joint_model(params, delta, taus)
            return np.sum((y - model) ** 2)

        de_result = differential_evolution(
            _objective,
            de_bounds,
            seed=42,
            maxiter=2000,
            tol=1e-10,
            polish=True,
            popsize=25,
        )
        bg_fit, amp_fit, delta0_fit, gamma_fit = de_result.x
        t2 = 1.0 / gamma_fit if gamma_fit > 1e-12 else np.nan
        fitted_curves = _joint_model(de_result.x, delta, taus)

        result["freq_offset"] = float(delta0_fit)
        result["contrast"] = float(abs(amp_fit))
        result["decay_rate"] = float(gamma_fit)
        result["t2_star"] = float(t2)
        result["fitted_curves"] = fitted_curves
        result["success"] = (
            np.isfinite(delta0_fit)
            and abs(amp_fit) > 1e-6
            and gamma_fit >= 0
        )

        _logger.debug(
            "Joint two-τ fit: δ₀=%.3f MHz, A₀=%.4f, γ=%.5f 1/ns, "
            "T2*=%.1f ns",
            delta0_fit * 1e-6,
            amp_fit,
            gamma_fit,
            t2 if np.isfinite(t2) else -1,
        )
    except Exception:
        _logger.debug("Joint two-τ DE fit failed", exc_info=True)

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit resonance detuning via joint two-τ differential evolution.

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
        qubit_names = [
            getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)
        ]

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
            f"A₀={contrast:.4f}, "
            f"γ={gamma:.5f} 1/ns, "
            f"T2*={t2:.0f} ns, "
            f"success={success}"
        )
        log_callable(msg)
