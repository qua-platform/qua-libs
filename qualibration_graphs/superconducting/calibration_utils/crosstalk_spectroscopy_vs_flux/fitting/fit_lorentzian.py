import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


# ------------------------------
# Lorentzian functions
# ------------------------------

def lorentzian(f, A, f0, gamma, B):
    """Full Lorentzian with all parameters free."""
    return A * (gamma ** 2 / ((f - f0) ** 2 + gamma ** 2)) + B


def estimate_lorentzian_initial_guess(freqs, mags):
    """Estimate initial guesses for A, f0, gamma, B."""
    A_guess = np.max(mags) - np.min(mags)
    f0_guess = freqs[np.argmax(mags)]
    B_guess = np.min(mags)

    # Estimate gamma as the min frequency step or fraction of total range
    df = np.diff(freqs)
    min_df = np.min(df[df > 0]) if np.any(df > 0) else 1e-6
    gamma_guess = min_df * 3  # or np.ptp(freqs) / 10

    return [A_guess, f0_guess, gamma_guess, B_guess]


def fit_lorentzian(detuning, magnitude):
    """Fit Lorentzian with all parameters free, return popt and errors."""
    try:
        p0 = estimate_lorentzian_initial_guess(detuning, magnitude)
        popt, pcov = curve_fit(lorentzian, detuning, magnitude, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return *popt, *perr
    except Exception:
        return None, None, None, None, None, None, None, None


# ------------------------------
# Restricted fit (only f0 free)
# ------------------------------

def make_lorentzian_fixed(A, gamma, B):
    """Factory for Lorentzian with only f0 free."""
    def lorentzian_only_f0(f, f0):
        return A * (gamma ** 2 / ((f - f0) ** 2 + gamma ** 2)) + B
    return lorentzian_only_f0


def fit_lorentzian_fixed(detuning, magnitude, A, gamma, B):
    """Fit only f0, keeping A, gamma, B fixed."""
    try:
        f0_guess = detuning[np.argmax(magnitude)]
        popt, pcov = curve_fit(
            make_lorentzian_fixed(A, gamma, B),
            detuning,
            magnitude,
            p0=[f0_guess],
        )
        f0 = popt[0]
        perr = np.sqrt(np.diag(pcov))
        return f0, perr[0]
    except Exception:
        return np.nan, np.nan


# ------------------------------
# Global parameter estimation
# ------------------------------

def estimate_global_parameters(da: xr.DataArray):
    """
    Estimate global A, gamma, B by averaging over flux_bias.
    Returns A, gamma, B.
    """
    (A_fit, f0_fit, gamma_fit, B_fit,
     A_fit_err, f0_fit_err, gamma_fit_err, B_fit_err) = xr.apply_ufunc(
        fit_lorentzian,
        da.detuning,
        da,
        input_core_dims=[["detuning"], ["detuning"]],
        output_core_dims=[[]]*8,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]*8,
    )

    total_err = (
        (A_fit_err / A_fit)**2 +
        (B_fit_err / B_fit)**2 +
        (gamma_fit_err / gamma_fit)**2 +
        (f0_fit_err / f0_fit) ** 2
    )
    # Weight according to prominence peak and scale by fit error
    weights = abs(da.std(dim="detuning") - da.std(dim="detuning").min()) * (1 / total_err)

    A_weighted_avg = (A_fit * weights).sum() / weights.sum()
    gamma_weighted_avg = (gamma_fit * weights).sum() / weights.sum()
    B_weighted_avg = (B_fit * weights).sum() / weights.sum()

    return A_weighted_avg.data, gamma_weighted_avg.data, B_weighted_avg.data


# ------------------------------
# Apply restricted fit across dataset
# ------------------------------

def fit_lorentzian_for_each_detuning_fixed(da: xr.DataArray):
    """
    Perform two-stage fitting:
    1. Fit averaged data to estimate A, gamma, B.
    2. Fit each cut with only f0 free.
    """
    # Step 1: estimate global A, gamma, B
    A, gamma, B = estimate_global_parameters(da)

    # Step 2: fit each detuning cut with only f0 free
    peak_freq, peak_freq_err = xr.apply_ufunc(
        fit_lorentzian_fixed,
        da.detuning,
        da,
        kwargs=dict(A=A, gamma=gamma, B=B),
        input_core_dims=[["detuning"], ["detuning"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    # Mask errored fits
    mask = (~peak_freq.isnull()) & (~peak_freq_err.isnull())

    # Mask values which are more than 10% out-of-bounds
    detuning_min = float(da.detuning.min().values)
    detuning_max = float(da.detuning.max().values)
    detuning_range = detuning_max - detuning_min

    lower_bound = detuning_min - 0.0 * detuning_range
    upper_bound = detuning_max + 0.0 * detuning_range

    mask &= (lower_bound < peak_freq) & (peak_freq < upper_bound)

    # Mask points whose error is greater than the whole window
    mask &= (peak_freq_err < (da.detuning.max() - da.detuning.min()) / 2)

    peak_freq = peak_freq.where(mask, drop=True)
    peak_freq_err = peak_freq_err.where(mask, drop=True)
    flux_bias = peak_freq.flux_bias.where(mask, drop=True)

    return peak_freq, peak_freq_err, flux_bias
