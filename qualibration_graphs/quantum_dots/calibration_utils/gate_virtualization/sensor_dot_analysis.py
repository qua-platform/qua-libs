"""Lorentzian fitting for sensor dot Coulomb peak tuning.

The standard Lorentzian used here is

    L(x) = (γ / 2π) / ((x − x0)² + (γ / 2)²) + offset

The inflection points — where the slope |dL/dx| is maximised, giving the
highest charge sensitivity — sit at  x0 ± γ / (2√3).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import curve_fit


class LorentzianFitResult(NamedTuple):
    """Container for the fitted Lorentzian parameters."""

    x0: float
    gamma: float
    amplitude: float
    offset: float
    optimal_voltage: float


def lorentzian(x: np.ndarray, x0: float, gamma: float, amplitude: float, offset: float) -> np.ndarray:
    """Lorentzian peak with free amplitude and offset.

    L(x) = amplitude * (γ/2)² / ((x − x0)² + (γ/2)²) + offset

    This parameterisation keeps the peak height = amplitude + offset at x = x0,
    which maps naturally to the sensor Coulomb peak signal.
    """
    half_gamma_sq = (gamma / 2) ** 2
    return amplitude * half_gamma_sq / ((x - x0) ** 2 + half_gamma_sq) + offset


def optimal_operating_point(x0: float, gamma: float, side: str = "right") -> float:
    """Return the voltage at the Lorentzian inflection point.

    The inflection points of the standard Lorentzian sit at
        x0 ± γ / (2√3)
    which is where |dL/dx| is maximised.
    """
    delta = gamma / (2 * np.sqrt(3))
    if side == "right":
        return x0 + delta
    elif side == "left":
        return x0 - delta
    raise ValueError(f"side must be 'left' or 'right', got '{side}'")


def fit_lorentzian(
    v: np.ndarray,
    signal: np.ndarray,
    side: str = "right",
) -> LorentzianFitResult:
    """Fit a Lorentzian peak to a 1D sensor sweep.

    Parameters
    ----------
    v : np.ndarray
        Voltage values (1D).
    signal : np.ndarray
        Measured sensor signal (1D, same length as *v*).
    side : str
        Which inflection point to use ('left' or 'right').

    Returns
    -------
    LorentzianFitResult
        Fitted parameters and the derived optimal voltage.
    """
    peak_idx = int(np.argmax(signal))
    x0_guess = float(v[peak_idx])
    amp_guess = float(signal[peak_idx] - np.median(signal))
    gamma_guess = float((v[-1] - v[0]) / 10)
    offset_guess = float(np.median(signal))

    popt, pcov = curve_fit(
        lorentzian,
        v,
        signal,
        p0=[x0_guess, gamma_guess, amp_guess, offset_guess],
        maxfev=10_000,
    )
    x0_fit, gamma_fit, amp_fit, offset_fit = popt
    gamma_fit = abs(gamma_fit)

    return LorentzianFitResult(
        x0=x0_fit,
        gamma=gamma_fit,
        amplitude=amp_fit,
        offset=offset_fit,
        optimal_voltage=optimal_operating_point(x0_fit, gamma_fit, side),
    )
