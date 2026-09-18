"""XEB fidelity analysis functions.

Provides metrics for evaluating the quality of quantum gates based on
cross-entropy benchmarking:

- **Linear XEB fidelity** (Arute et al. 2019):
    F_XEB = d ⟨p_ideal⟩ − 1
  where d is the Hilbert-space dimension and ⟨p_ideal⟩ is the mean ideal
  probability weighted by the measured distribution.

- **Log XEB fidelity** (cross-entropy based):
    F_log = 1 + ⟨log(d · p_ideal)⟩ / log(d)
  (average is over measured shots)

- **Speckle purity**:
    Pu = d ⟨p_ideal²⟩ − 1
  A quantity that decays as r_pu^depth; for a depolarizing channel,
  r_pu ≈ r_xeb (layer fidelity).

Both fidelities are fitted to an exponential decay
    F(depth) = a · r^depth
to extract the layer fidelity *r* and SPAM parameter *a*.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


def _exponential_decay(d: np.ndarray, a: float, r: float) -> np.ndarray:
    """Model: a * r^d."""
    return a * r**d


def _fit_decay(
    depths: np.ndarray, values: np.ndarray
) -> Tuple[float, float, float, float]:
    """Fit exponential decay and return (a, r, a_err, r_err).

    Falls back to NaN if fitting fails.
    """
    try:
        popt, pcov = curve_fit(
            _exponential_decay,
            depths.astype(float),
            values.astype(float),
            p0=[1.0, 0.99],
            bounds=([0, 0], [np.inf, 1.0]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt[0], popt[1], perr[0], perr[1]
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.nan


def calc_linear_xeb_fidelity(
    measured_probs: np.ndarray,
    ideal_probs: np.ndarray,
    dim: int = 2,
) -> np.ndarray:
    """Compute the linear XEB fidelity for each (sequence, depth).

    Parameters
    ----------
    measured_probs : ndarray, shape (..., dim)
        Measured probability distribution (from shot counting).
    ideal_probs : ndarray, shape (..., dim)
        Ideal (noiseless) probability distribution.
    dim : int
        Hilbert-space dimension (2 for 1Q, 4 for 2Q).

    Returns
    -------
    ndarray, shape (...)
        Linear XEB fidelity per (sequence, depth).
    """
    return dim * np.sum(measured_probs * ideal_probs, axis=-1) - 1


def calc_log_xeb_fidelity(
    measured_probs: np.ndarray,
    ideal_probs: np.ndarray,
    dim: int = 2,
) -> np.ndarray:
    """Compute the logarithmic XEB fidelity for each (sequence, depth).

    Parameters
    ----------
    measured_probs : ndarray, shape (..., dim)
        Measured probability distribution.
    ideal_probs : ndarray, shape (..., dim)
        Ideal probability distribution.
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    ndarray, shape (...)
        Log XEB fidelity per (sequence, depth).
    """
    eps = 1e-30
    log_vals = np.log(dim * np.clip(ideal_probs, eps, None))
    return 1 + np.sum(measured_probs * log_vals, axis=-1) / np.log(dim)


def calc_purity(
    measured_probs: np.ndarray,
    dim: int = 2,
) -> np.ndarray:
    """Compute the speckle purity for each (sequence, depth).

    Purity = dim * mean_over_sequences(sum(p_measured^2)) - 1.
    For a depolarizing channel, the purity decay rate approximates the
    layer fidelity.

    Parameters
    ----------
    measured_probs : ndarray, shape (..., dim)
        Measured probability distribution.
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    ndarray, shape (...)
        Speckle purity per (sequence, depth).
    """
    return dim * np.sum(measured_probs**2, axis=-1) - 1


def log_xeb_results(
    depths: np.ndarray,
    linear_fid: np.ndarray,
    log_fid: np.ndarray,
    purity: np.ndarray,
    label: str = "",
) -> dict:
    """Fit decay curves and package XEB results into a dict.

    Parameters
    ----------
    depths : ndarray of int
        Circuit depths.
    linear_fid : ndarray, shape (n_sequences, n_depths)
        Linear XEB fidelity per (sequence, depth).
    log_fid : ndarray, shape (n_sequences, n_depths)
        Log XEB fidelity per (sequence, depth).
    purity : ndarray, shape (n_sequences, n_depths)
        Speckle purity per (sequence, depth).
    label : str
        Identifying label (e.g. qubit name or qubit pair).

    Returns
    -------
    dict
        Dictionary with keys: label, depths, mean fidelities,
        fitted parameters and uncertainties.
    """
    mean_lin = np.mean(linear_fid, axis=0)
    mean_log = np.mean(log_fid, axis=0)
    mean_pur = np.mean(purity, axis=0)

    a_lin, r_lin, a_lin_err, r_lin_err = _fit_decay(depths, mean_lin)
    a_log, r_log, a_log_err, r_log_err = _fit_decay(depths, mean_log)
    a_pur, r_pur, a_pur_err, r_pur_err = _fit_decay(
        depths, np.sqrt(np.clip(mean_pur, 0, None))
    )

    return {
        "label": label,
        "depths": depths,
        "linear_fidelity_mean": mean_lin,
        "log_fidelity_mean": mean_log,
        "purity_mean": mean_pur,
        "linear_fit": {"a": a_lin, "r": r_lin, "a_err": a_lin_err, "r_err": r_lin_err},
        "log_fit": {"a": a_log, "r": r_log, "a_err": a_log_err, "r_err": r_log_err},
        "purity_fit": {"a": a_pur, "r": r_pur, "a_err": a_pur_err, "r_err": r_pur_err},
    }
