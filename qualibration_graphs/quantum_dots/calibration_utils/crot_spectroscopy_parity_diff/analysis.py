"""CROT spectroscopy parity-diff analysis.

The CROT spectroscopy measures the target qubit's resonance frequency as a
function of the exchange voltage, with and without an x180 pulse on the
control qubit.  When the control is in |↓⟩ the target resonates at f_↓;
when flipped to |↑⟩ it resonates at f_↑.  The exchange coupling is
J = |f_↑ − f_↓|.

For each exchange voltage slice the parity-difference signal vs frequency
is fitted with a Lorentzian to locate the resonance peak.  Comparing the
peak positions between the ``control_x180 = False`` and ``True`` datasets
gives J(V_exchange).

**Fitting strategy**

Each 1-D frequency trace (at fixed exchange voltage and control state) is
fitted to a Lorentzian:

    P(f) = offset + A / (1 + ((f − f0) / (Γ/2))²)

using ``scipy.optimize.curve_fit`` with sensible initial guesses derived
from the data (peak location from argmax, width from FWHM estimate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
from scipy.optimize import curve_fit

import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class LorentzianFitResult:
    """Fitted parameters for a single Lorentzian trace."""

    f0: float = 0.0
    """Resonance frequency (Hz)."""
    amplitude: float = 0.0
    """Peak amplitude."""
    gamma: float = 0.0
    """Full width at half maximum (Hz)."""
    offset: float = 0.0
    """Baseline offset."""
    success: bool = False
    """Whether the fit converged to a physically sensible result."""


@dataclass
class ExchangeModelFit:
    """Fitted parameters for J(V) = J_0 * exp((V - V_ref) / lever_arm)."""

    J_0: float = 0.0
    """Exchange coupling at V_ref (Hz)."""
    V_ref: float = 0.0
    """Reference voltage (V)."""
    lever_arm: float = 0.0
    """Lever arm (V)."""
    success: bool = False
    """Whether the exponential fit converged."""


@dataclass
class CROTFitResult:
    """Aggregated fit result for one qubit pair."""

    exchange_coupling_J: float = 0.0
    """Exchange coupling |f_↑ − f_↓| in Hz, evaluated at the exchange voltage
    where the splitting is best resolved."""
    crot_frequency_down: float = 0.0
    """Target qubit resonance when control is |↓⟩ (Hz)."""
    crot_frequency_up: float = 0.0
    """Target qubit resonance when control is |↑⟩ (Hz)."""
    optimal_exchange_idx: int = 0
    """Index into the exchange array where J is best resolved."""
    exchange_model: ExchangeModelFit | None = None
    """Fitted exponential exchange model J(V)."""
    success: bool = False
    """Whether the extraction succeeded."""


def _lorentzian(f: np.ndarray, f0: float, amplitude: float, gamma: float, offset: float) -> np.ndarray:
    return offset + amplitude / (1.0 + ((f - f0) / (gamma / 2.0)) ** 2)


def _fit_lorentzian(freqs: np.ndarray, signal: np.ndarray) -> LorentzianFitResult:
    """Fit a Lorentzian to a 1-D frequency trace."""
    result = LorentzianFitResult()
    if len(freqs) < 5:
        return result

    try:
        peak_idx = int(np.nanargmax(signal))
        f0_guess = freqs[peak_idx]
        amp_guess = float(np.nanmax(signal) - np.nanmin(signal))
        offset_guess = float(np.nanmin(signal))
        half_max = offset_guess + amp_guess / 2.0
        above = np.where(signal > half_max)[0]
        gamma_guess = float(freqs[above[-1]] - freqs[above[0]]) if len(above) > 1 else float(np.ptp(freqs)) / 10.0
        gamma_guess = max(gamma_guess, float(np.min(np.abs(np.diff(freqs)))))

        popt, _ = curve_fit(
            _lorentzian,
            freqs,
            signal,
            p0=[f0_guess, amp_guess, gamma_guess, offset_guess],
            maxfev=5000,
        )
        f0_fit, amp_fit, gamma_fit, offset_fit = popt
        result.f0 = float(f0_fit)
        result.amplitude = float(amp_fit)
        result.gamma = float(abs(gamma_fit))
        result.offset = float(offset_fit)
        result.success = bool(np.isfinite(f0_fit) and abs(gamma_fit) > 0)
    except (RuntimeError, ValueError):
        pass

    return result


def _fit_exchange_model(
    exchange_voltages: np.ndarray,
    j_values_hz: np.ndarray,
) -> ExchangeModelFit:
    """Fit J(V) = J_0 * exp((V - V_ref) / lever_arm) to the extracted coupling."""
    result = ExchangeModelFit()
    valid = np.isfinite(j_values_hz) & (j_values_hz > 0)
    if np.sum(valid) < 3:
        return result

    v = exchange_voltages[valid]
    j = j_values_hz[valid]

    try:
        log_j = np.log(j)
        slope, intercept = np.polyfit(v, log_j, 1)
        lever_arm_guess = 1.0 / slope if abs(slope) > 1e-12 else 0.1
        j0_guess = np.exp(intercept)

        def _model(v_arr, j0, v_ref, lam):
            return j0 * np.exp((v_arr - v_ref) / lam)

        popt, _ = curve_fit(
            _model,
            v,
            j,
            p0=[j0_guess, 0.0, lever_arm_guess],
            maxfev=5000,
        )
        result.J_0 = float(popt[0])
        result.V_ref = float(popt[1])
        result.lever_arm = float(popt[2])
        result.success = bool(np.isfinite(popt).all() and result.J_0 > 0 and result.lever_arm != 0)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass

    return result


def _extract_crot_params(
    freqs: np.ndarray,
    exchange_voltages: np.ndarray,
    pdiff_no_x180: np.ndarray,
    pdiff_with_x180: np.ndarray,
) -> tuple[CROTFitResult, np.ndarray, np.ndarray]:
    """Extract CROT parameters from two 2-D parity-diff maps.

    Parameters
    ----------
    freqs : 1-D array  (n_freq,)
    exchange_voltages : 1-D array  (n_exchange,)
    pdiff_no_x180 : 2-D array  (n_exchange, n_freq)
    pdiff_with_x180 : 2-D array  (n_exchange, n_freq)

    Returns
    -------
    result : CROTFitResult
    f0_down : 1-D array of peak frequencies per exchange point (no x180)
    f0_up : 1-D array of peak frequencies per exchange point (with x180)
    """
    n_exchange = pdiff_no_x180.shape[0]
    f0_down = np.full(n_exchange, np.nan)
    f0_up = np.full(n_exchange, np.nan)

    for i in range(n_exchange):
        fit_down = _fit_lorentzian(freqs, pdiff_no_x180[i])
        fit_up = _fit_lorentzian(freqs, pdiff_with_x180[i])
        if fit_down.success:
            f0_down[i] = fit_down.f0
        if fit_up.success:
            f0_up[i] = fit_up.f0

    splitting = np.abs(f0_up - f0_down)
    valid = np.isfinite(splitting)

    result = CROTFitResult()
    if not np.any(valid):
        return result, f0_down, f0_up

    best_idx = int(np.nanargmax(splitting))
    result.optimal_exchange_idx = best_idx
    result.exchange_coupling_J = float(splitting[best_idx])
    result.crot_frequency_down = float(f0_down[best_idx])
    result.crot_frequency_up = float(f0_up[best_idx])
    result.success = bool(np.isfinite(result.exchange_coupling_J) and result.exchange_coupling_J > 0)

    result.exchange_model = _fit_exchange_model(exchange_voltages, splitting)

    return result, f0_down, f0_up


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run CROT spectroscopy analysis for every qubit pair.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with coordinates ``control_x180``, ``exchange``,
        ``esr_frequency`` and data variables ``pdiff_<pair_name>``.
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).

    Returns
    -------
    ds_fit : xr.Dataset
        Fitted peak positions (``f0_down_<pair>``, ``f0_up_<pair>``) vs exchange.
    fit_results : dict
        ``{pair_name: {exchange_coupling_J, crot_frequency_down,
        crot_frequency_up, optimal_exchange_idx, success}}``.
    """
    freqs = ds_raw.coords["esr_frequency"].values.astype(np.float64)
    exchange = ds_raw.coords["exchange"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"pdiff_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No pdiff variable for pair %s — skipping.", qp.name)
            fit_results[qp.name] = asdict(CROTFitResult())
            continue

        pdiff = ds_raw[var_name].values.astype(np.float64)
        # pdiff shape: (n_control_x180, n_exchange, n_esr_frequency)
        pdiff_no_x180 = pdiff[0]
        pdiff_with_x180 = pdiff[1]

        result, f0_down, f0_up = _extract_crot_params(freqs, exchange, pdiff_no_x180, pdiff_with_x180)
        result_dict = asdict(result)
        if result.exchange_model is not None:
            result_dict["exchange_model"] = asdict(result.exchange_model)
        fit_results[qp.name] = result_dict

        fit_vars[f"f0_down_{qp.name}"] = xr.DataArray(
            f0_down, dims=["exchange"], attrs={"long_name": "f_↓", "units": "Hz"}
        )
        fit_vars[f"f0_up_{qp.name}"] = xr.DataArray(f0_up, dims=["exchange"], attrs={"long_name": "f_↑", "units": "Hz"})

    ds_fit = xr.Dataset(fit_vars, coords={"exchange": exchange})
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log CROT spectroscopy fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        j_mhz = r["exchange_coupling_J"] / 1e6
        msg = (
            f"  {name}: [{status}] J = {j_mhz:.3f} MHz, "
            f"f_↓ = {r['crot_frequency_down'] / 1e6:.3f} MHz, "
            f"f_↑ = {r['crot_frequency_up'] / 1e6:.3f} MHz"
        )
        _log(msg)
        em = r.get("exchange_model")
        if em and em.get("success"):
            _log(
                f"    Exchange model: J_0 = {em['J_0'] / 1e6:.3f} MHz, "
                f"V_ref = {em['V_ref'] * 1e3:.1f} mV, "
                f"lever_arm = {em['lever_arm'] * 1e3:.1f} mV"
            )
