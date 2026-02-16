"""1-D Ramsey detuning-sweep analysis via linear cosine decomposition.

This module analyses data from a Ramsey experiment where the **idle time
τ is fixed** and the **drive-frequency detuning δ** is swept.

At fixed τ the parity-difference signal is a pure cosine in detuning:

.. math::

    P(\\delta) = \\mathrm{bg}
        + C\\,\\cos(2\\pi f\\,\\delta)
        + S\\,\\sin(2\\pi f\\,\\delta)

where *f* = τ_ns × 10⁻⁹ (Hz⁻¹) is **known** — it is not a free
parameter.  The three unknowns (bg, C, S) are solved exactly by
ordinary least squares (``np.linalg.lstsq``).

From the solution the resonance position is

.. math::

    \\delta_0 = -\\frac{\\mathrm{atan2}(S,\\,C)}{2\\pi f}

and the contrast is *A* = √(C² + S²).

Because the oscillation period is dictated by the known idle time,
there is no non-linear optimisation — the solution is globally
optimal in one matrix solve.

Extracted quantities
--------------------
* **freq_offset** — resonance detuning δ₀ (Hz), the correction to
  apply to the qubit intermediate frequency.
* **contrast** — oscillation contrast *A*.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a detuning-swept Ramsey measurement.

    Attributes
    ----------
    freq_offset : float
        Resonance detuning δ₀ (Hz) — the correction to subtract from
        the qubit intermediate frequency so the drive sits on resonance.
    contrast : float
        Oscillation contrast A = √(C² + S²).
    success : bool
        ``True`` if the linear solve produced a finite result with
        positive contrast.
    """

    freq_offset: float
    contrast: float
    success: bool


# ── Single-qubit analysis ────────────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff_1d: np.ndarray,
    detuning_hz: np.ndarray,
    tau_ns: float,
) -> Dict[str, Any]:
    r"""Fit a single qubit's detuning-swept Ramsey trace.

    Builds the design matrix ``[1, cos(2πfδ), sin(2πfδ)]`` with
    *f* = τ_ns × 1e-9 and solves for (bg, C, S) via least squares.
    Then computes:

    * contrast  A  = √(C² + S²)
    * resonance δ₀ = −atan2(S, C) / (2πf)

    Parameters
    ----------
    pdiff_1d : 1-D array (n_det,)
        Parity-difference values.
    detuning_hz : 1-D array (n_det,)
        Detuning values in Hz.
    tau_ns : float
        Fixed idle time in nanoseconds.

    Returns
    -------
    dict
        Keys: ``freq_offset`` (Hz), ``contrast``, ``success``,
        ``fitted_curve`` (1-D array), ``bg``, ``C``, ``S``.
    """
    y = np.asarray(pdiff_1d, dtype=float)
    delta = np.asarray(detuning_hz, dtype=float)
    f = tau_ns * 1e-9  # oscillation "frequency" in Hz⁻¹

    phase = 2.0 * np.pi * f * delta
    # Design matrix: [1, cos(phase), sin(phase)]
    A_mat = np.column_stack([np.ones_like(delta), np.cos(phase), np.sin(phase)])

    result: Dict[str, Any] = {
        "freq_offset": 0.0,
        "contrast": 0.0,
        "success": False,
        "fitted_curve": np.full_like(y, np.nan),
        "bg": np.nan,
        "C": np.nan,
        "S": np.nan,
    }

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        bg, C, S = coeffs

        contrast = float(np.hypot(C, S))
        freq_offset = float(-np.arctan2(S, C) / (2.0 * np.pi * f))
        fitted_curve = A_mat @ coeffs

        result["bg"] = float(bg)
        result["C"] = float(C)
        result["S"] = float(S)
        result["contrast"] = contrast
        result["freq_offset"] = freq_offset
        result["fitted_curve"] = fitted_curve
        result["success"] = np.isfinite(freq_offset) and contrast > 0
    except Exception:
        _logger.debug("Linear cosine decomposition failed", exc_info=True)

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit resonance detuning for each qubit via linear cosine decomposition.

    Expects a 1-D dataset with coordinate ``detuning`` (Hz), with data
    variables ``pdiff_<qubit>`` of shape ``(n_det,)``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data.
    node : QualibrationNode
        Calibration node (provides qubit list and ``idle_time_ns``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name → dict of :class:`FitParameters` fields plus
        ``_diag`` with raw solver outputs for plotting.
    """
    qubits = node.namespace["qubits"]
    tau_ns = float(node.parameters.idle_time_ns)
    detuning_hz = np.asarray(ds.detuning.values, dtype=float)

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
            fp = FitParameters(freq_offset=0.0, contrast=0.0, success=False)
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        raw = _analyse_single_qubit(pdiff, detuning_hz, tau_ns)

        fp = FitParameters(
            freq_offset=raw["freq_offset"],
            contrast=raw["contrast"],
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
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"δ₀={freq_off_mhz:.4f} MHz, "
            f"contrast={contrast:.4f}, "
            f"success={success}"
        )
        log_callable(msg)
