"""Analysis for the T2*-versus-flux (Ramsey dephasing) node.

A decaying-oscillation ``a*exp(-t*decay)*cos(2*pi*f*t + phi) + offset`` is fitted along
the idle-time axis for every flux-bias point, and T2*(flux) = 1/decay is extracted from
the envelope. The frequency ``f`` (the artificial detuning) is not used here -- only the
decay envelope matters for T2*.

Each (qubit, flux_bias) curve is fitted with a local, guarded ``curve_fit`` wrapped in
``xr.apply_ufunc(vectorize=True)`` (see ``_fit_oscillation_vs_flux``). We deliberately do
NOT call the shared ``fit_oscillation_decay_exp``: on a ``curve_fit`` RuntimeError it pops a
BLOCKING ``plt.show()``, and a flux sweep routinely produces unfittable far-detuned /
low-contrast slices that would hang a headless/batch node run. The local fitter returns an
all-NaN row instead, which the R^2 and relative-error valid-gates then discard.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import oscillation_decay_exp, guess

# Minimum goodness-of-fit (R^2) for a flux point's Ramsey fit to count as valid.
# Clean fringes give R^2 ~ 0.99; noise-only points give R^2 ~ 0, so this cleanly
# rejects flux points where the decaying-oscillation fit latched onto noise.
_R2_MIN = 0.5


@dataclass
class T2StarVsFluxFit:
    """Stores the T2*-versus-flux summary for a single qubit."""

    t2_star_max: float
    """Longest fitted T2* over the flux sweep, in seconds."""
    t2_star_max_error: float
    """Uncertainty on ``t2_star_max``, in seconds."""
    flux_at_max: float
    """Flux bias (V) at which the longest T2* is observed."""
    num_valid_flux: int
    """Number of flux points that produced a physical (finite, positive) T2*."""
    success: bool
    """Whether enough flux points were successfully fitted."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the per-qubit T2*-versus-flux summary."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        status = "SUCCESS" if r["success"] else "FAIL"
        log_callable(
            f"T2* vs flux for {q} : max T2* = {1e6 * r['t2_star_max']:.2f} +/- "
            f"{1e6 * r['t2_star_max_error']:.2f} us at flux = {r['flux_at_max']:.4f} V "
            f"({r['num_valid_flux']} valid flux points) --> {status}!"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to voltage if state discrimination is not used."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, T2StarVsFluxFit]]:
    """Fit the Ramsey decay for every flux-bias point and extract the per-qubit summary."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    data = ds[signal_name]
    # Fit each (qubit, flux_bias) Ramsey curve with a local guarded curve_fit (see
    # _fit_oscillation_vs_flux). We do NOT call the shared fit_oscillation_decay_exp because its
    # internal apply_fit pops a BLOCKING plt.show() on any curve_fit RuntimeError -- a flux sweep
    # routinely produces unfittable far-detuned/low-contrast slices, which would hang a headless run.
    fit_data = _fit_oscillation_vs_flux(data, "idle_time")
    # Goodness-of-fit (R^2) per (qubit, flux_bias). A decaying oscillation can spuriously
    # fit pure noise, so we additionally reject flux points whose reconstructed fit poorly
    # matches the data (the rel-error gate alone is insufficient for the oscillation case).
    fitted = oscillation_decay_exp(
        data.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )
    ss_res = ((data - fitted) ** 2).sum("idle_time")
    ss_tot = ((data - data.mean("idle_time")) ** 2).sum("idle_time")
    r2 = 1.0 - ss_res / ss_tot
    ds_fit = xr.merge([ds, fit_data])
    ds_fit["fit_r2"] = r2
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _fit_oscillation_vs_flux(da: xr.DataArray, time_dim: str = "idle_time") -> xr.DataArray:
    """Fit ``a*exp(-t*decay)*cos(2*pi*f*t + phi) + offset`` along ``time_dim`` for each
    (qubit, flux_bias), vectorised and robust.

    Returns a DataArray named ``fit_data`` with a ``fit_vals`` dim holding
    ``[a, f, phi, offset, decay, decay_var]`` (decay_var = variance of the fitted decay).
    Any curve that cannot be fitted yields an all-NaN row instead of raising or, crucially,
    reaching the shared fitter's blocking ``plt.show()`` path."""
    t = da[time_dim]

    def _fit_one(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 6:
            return np.array([np.nan] * 6)
        x, y = x[finite], y[finite]
        span = max(x.max() - x.min(), 1.0)
        # Initial guesses (reuse the shared guess helpers used by fit_oscillation_decay_exp).
        a0 = float((np.max(y) - np.min(y)) / 2) or 1.0
        offset0 = float(np.mean(y))
        try:
            f0 = float(guess.frequency(x, y))
        except Exception:
            f0 = 1.0 / span
        try:
            decay0 = float(guess.oscillation_exp_decay(x, y))
        except Exception:
            decay0 = 1.0 / span
        if not np.isfinite(f0) or f0 <= 0:
            f0 = 1.0 / span
        if not np.isfinite(decay0) or decay0 == 0:
            decay0 = 1.0 / span
        try:
            popt, pcov = curve_fit(
                oscillation_decay_exp, x, y, p0=[a0, f0, 0.0, offset0, decay0], maxfev=10000
            )
            decay_var = float(np.abs(np.diag(pcov)[4]))
            return np.array([popt[0], popt[1], popt[2], popt[3], popt[4], decay_var])
        except Exception:
            return np.array([np.nan] * 6)

    fit = xr.apply_ufunc(
        _fit_one,
        t,
        da,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit = fit.assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset", "decay", "decay_var"]))
    return fit.rename("fit_data")


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, T2StarVsFluxFit]]:
    """Turn fitted decay rates into T2*(flux) and per-qubit summaries."""
    # Model is a*exp(-t*decay)*cos(...)+offset, so decay > 0 for a real decay and T2* = 1/decay.
    decay = ds_fit.fit_data.sel(fit_vals="decay")
    decay_var = ds_fit.fit_data.sel(fit_vals="decay_var")

    tau = 1.0 / decay
    tau_error = np.abs(tau) * (np.sqrt(np.abs(decay_var)) / np.abs(decay))
    # Keep only physical, well-constrained dephasing times: finite & positive, above the
    # ~1-clock-cycle (16 ns) floor, and with a relative error < 1. This discards noise-only
    # / unconverged fits (decay <= 0 or ~0 -> runaway / negative tau) so they are never
    # counted valid nor written to qubit.extras.
    rel_err = np.abs(tau_error / tau)
    r2 = ds_fit["fit_r2"] if "fit_r2" in ds_fit else xr.ones_like(decay)
    valid = (
        np.isfinite(tau)
        & (tau > 16)
        & np.isfinite(tau_error)
        & (rel_err < 1)
        & (r2 >= _R2_MIN)
    )
    tau = tau.where(valid)
    tau_error = tau_error.where(valid)

    ds_fit["T2_star"] = tau
    ds_fit["T2_star"].attrs = {"long_name": "T2*", "units": "ns"}
    ds_fit["T2_star_error"] = tau_error
    ds_fit["T2_star_error"].attrs = {"long_name": "T2* error", "units": "ns"}

    n_flux = ds_fit.sizes.get("flux_bias", 0)
    min_valid = max(3, n_flux // 2)

    fit_results = {}
    for q in ds_fit.qubit.values:
        tau_q = ds_fit["T2_star"].sel(qubit=q)
        err_q = ds_fit["T2_star_error"].sel(qubit=q)
        num_valid = int(np.isfinite(tau_q).sum())
        if num_valid == 0:
            fit_results[str(q)] = T2StarVsFluxFit(
                t2_star_max=float("nan"),
                t2_star_max_error=float("nan"),
                flux_at_max=float("nan"),
                num_valid_flux=0,
                success=False,
            )
            continue
        flux_at_max = float(tau_q.idxmax(dim="flux_bias", skipna=True).values)
        t2_max_ns = float(tau_q.max(skipna=True).values)
        t2_max_err_ns = float(err_q.sel(flux_bias=flux_at_max).values)
        fit_results[str(q)] = T2StarVsFluxFit(
            t2_star_max=1e-9 * t2_max_ns,
            t2_star_max_error=1e-9 * t2_max_err_ns,
            flux_at_max=flux_at_max,
            num_valid_flux=num_valid,
            success=bool(num_valid >= min_valid),
        )
    return ds_fit, fit_results
