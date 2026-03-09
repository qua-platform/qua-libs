"""PSB sweep-detuning analysis: sigmoid fit to the charge-sensor response.

This module fits a logistic (sigmoid) step to the averaged charge-sensor
signal measured while sweeping the measurement detuning through the
Pauli Spin Blockade (PSB) region:

.. math::

    S(\\varepsilon) = \\mathrm{baseline}
        + \\frac{\\mathrm{contrast}}
               {1 + \\exp\\bigl(-k\\,(\\varepsilon - \\varepsilon_0)\\bigr)}

In the PSB window, triplet states are blocked from tunnelling, producing
a different charge-sensor response than singlet states which tunnel freely.
As detuning moves beyond the PSB window, spin relaxation allows the triplet
to decay and the sensor response transitions to the unblocked level.

Fitting method
~~~~~~~~~~~~~~

A **profiled differential-evolution** search is used: DE optimises
over the two non-linear parameters (steepness *k*, transition detuning
*ε₀*), while the two linear parameters (baseline, contrast) are solved
exactly via ``np.linalg.lstsq`` at each candidate.

Extracted quantities
--------------------
* **transition_detuning** — detuning at which the PSB signature transitions (V).
* **contrast** — signal change between blocked and unblocked plateaux.
* **baseline** — signal level in the PSB (blocked) plateau.
* **steepness** — sharpness of the transition (1/V).
* **optimal_readout_detuning** — suggested readout detuning inside the PSB region (V).
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


@dataclass
class FitParameters:
    """Extracted parameters from a PSB sweep-detuning measurement.

    Attributes
    ----------
    transition_detuning : float
        Detuning at the midpoint of the PSB transition (V).
    contrast : float
        Signal difference between blocked and unblocked plateaux.
    baseline : float
        Signal level in the blocked plateau.
    steepness : float
        Sharpness of the transition (1/V).
    optimal_readout_detuning : float
        Suggested readout detuning inside the PSB region (V).
    success : bool
        ``True`` if the fit converged and the transition is physical.
    """

    transition_detuning: float
    contrast: float
    baseline: float
    steepness: float
    optimal_readout_detuning: float
    success: bool


def _fit_single_sensor(
    signal: np.ndarray,
    detuning: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit a sigmoid step to a single sensor's PSB sweep data.

    Uses profiled DE: 2-D search over (steepness, ε₀) with linear
    solve for (baseline, contrast) at each candidate.

    Parameters
    ----------
    signal : 1-D array (n_det,)
        Sensor signal values (e.g. IQ amplitude).
    detuning : 1-D array (n_det,)
        Detuning values in volts.

    Returns
    -------
    dict
        ``transition_detuning`` (V), ``contrast``, ``baseline``,
        ``steepness`` (1/V), ``optimal_readout_detuning`` (V),
        ``fitted_curve``, ``success``.
    """
    y = np.asarray(signal, dtype=float)
    x = np.asarray(detuning, dtype=float)
    n = len(x)
    x_span = float(x[-1] - x[0]) if n > 1 else 1.0

    result: Dict[str, Any] = {
        "transition_detuning": np.nan,
        "contrast": 0.0,
        "baseline": np.nan,
        "steepness": np.nan,
        "optimal_readout_detuning": np.nan,
        "fitted_curve": np.full_like(y, np.nan),
        "success": False,
    }

    if n < 4:
        return result

    # DE bounds for steepness k (positive; direction handled by contrast sign)
    k_lo = 1.0 / max(abs(x_span), 1e-9)
    k_hi = max(100.0 / max(abs(x_span), 1e-9), k_lo * 10.0)

    # DE bounds for transition detuning x0 (within the data range ± 10%)
    margin = 0.1 * abs(x_span)
    x0_lo = float(x[0]) - margin
    x0_hi = float(x[-1]) + margin

    def _profile_residual(params):
        k, x0 = params
        sigma = 1.0 / (1.0 + np.exp(-k * (x - x0)))
        A_mat = np.column_stack([np.ones(n), sigma])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        return float(np.sum((y - A_mat @ coeffs) ** 2))

    try:
        de_result = differential_evolution(
            _profile_residual,
            bounds=[(k_lo, k_hi), (x0_lo, x0_hi)],
            seed=42,
            maxiter=1000,
            tol=1e-12,
            polish=True,
            popsize=30,
        )
        k_best, x0_best = float(de_result.x[0]), float(de_result.x[1])

        # Solve linear system at best (k, x0)
        sigma = 1.0 / (1.0 + np.exp(-k_best * (x - x0_best)))
        A_mat = np.column_stack([np.ones(n), sigma])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        baseline, contrast = float(coeffs[0]), float(coeffs[1])

        fitted_curve = A_mat @ coeffs

        # Optimal readout: 90% into the PSB plateau (lower detuning side)
        optimal_readout = x0_best - np.log(9) / k_best
        optimal_readout = float(np.clip(optimal_readout, x[0], x[-1]))

        result["transition_detuning"] = x0_best
        result["contrast"] = contrast
        result["baseline"] = baseline
        result["steepness"] = k_best
        result["optimal_readout_detuning"] = optimal_readout
        result["fitted_curve"] = fitted_curve
        result["success"] = np.isfinite(x0_best) and x[0] <= x0_best <= x[-1] and abs(contrast) > 1e-6

        _logger.debug(
            "PSB fit: ε₀=%.4f V, contrast=%.4f, baseline=%.4f, k=%.1f 1/V",
            x0_best,
            contrast,
            baseline,
            k_best,
        )
    except Exception:
        _logger.debug("PSB sigmoid fit failed", exc_info=True)

    return result


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit PSB sigmoid for each dot-pair / sensor combination.

    Expects a 1-D dataset with coordinate ``detuning`` (V) and data
    variables ``I_<pair>_sensor_<i>`` / ``Q_<pair>_sensor_<i>``
    of shape ``(n_detuning,)``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data.
    node : QualibrationNode
        Calibration node (provides dot-pair list via namespace).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        sensor key → dict of :class:`FitParameters` fields plus
        ``_diag`` with raw solver outputs for plotting.
    """
    dot_pairs = node.namespace["dot_pairs"]
    detuning = np.asarray(ds.detuning.values, dtype=float)

    i_vars = sorted([v for v in ds.data_vars if v.startswith("I_")])
    sensor_keys = [v.replace("I_", "") for v in i_vars]
    if not sensor_keys:
        sensor_keys = [f"{pair.name}_sensor_0" for pair in dot_pairs]

    fit_results: Dict[str, Dict[str, Any]] = {}

    for key in sensor_keys:
        i_var = f"I_{key}"
        q_var = f"Q_{key}"

        if i_var not in ds.data_vars:
            fp = FitParameters(
                transition_detuning=np.nan,
                contrast=0.0,
                baseline=np.nan,
                steepness=np.nan,
                optimal_readout_detuning=np.nan,
                success=False,
            )
            fit_results[key] = asdict(fp)
            continue

        I = np.asarray(ds[i_var].values, dtype=float)
        Q = np.asarray(ds[q_var].values, dtype=float) if q_var in ds.data_vars else np.zeros_like(I)
        amplitude = np.sqrt(I**2 + Q**2)

        raw = _fit_single_sensor(amplitude, detuning)

        fp = FitParameters(
            transition_detuning=raw["transition_detuning"],
            contrast=raw["contrast"],
            baseline=raw["baseline"],
            steepness=raw["steepness"],
            optimal_readout_detuning=raw["optimal_readout_detuning"],
            success=raw["success"],
        )
        fit_results[key] = asdict(fp)
        fit_results[key]["_diag"] = raw

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted PSB transition parameters for all sensors.

    Parameters
    ----------
    fit_results : dict
        Sensor key → fit-result dict (as returned by :func:`fit_raw_data`).
    log_callable : callable, optional
        Logging function; defaults to ``_logger.info``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for key, r in fit_results.items():
        eps0 = r.get("transition_detuning", np.nan)
        contrast = r.get("contrast", 0)
        baseline = r.get("baseline", np.nan)
        opt = r.get("optimal_readout_detuning", np.nan)
        success = r.get("success", False)
        msg = (
            f"Results for {key}: "
            f"ε₀={eps0 * 1e3:.2f} mV, "
            f"contrast={contrast:.4f}, "
            f"baseline={baseline:.4f}, "
            f"optimal_readout={opt * 1e3:.2f} mV, "
            f"success={success}"
        )
        log_callable(msg)
