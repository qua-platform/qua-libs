import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from sklearn import linear_model


def linear(phi, m, c):
    return m * phi + c


def lorentzian(f, A, f0, gamma, B):
    """Full Lorentzian with all parameters free."""
    return A * (gamma ** 2 / ((f - f0) ** 2 + gamma ** 2)) + B


def estimate_lorentzian_initial_guess(freqs, mags):
    """Estimate initial guesses for A, f0, gamma, B."""
    A_guess = np.max(mags) - np.min(mags)
    f0_guess = freqs[np.argmax(mags)]
    B_guess = np.min(mags)

    df = np.diff(freqs)
    min_df = np.min(df[df > 0]) if np.any(df > 0) else 1e-6
    gamma_guess = min_df * 3

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


def estimate_global_parameters(da: xr.DataArray):
    """Estimate global A, gamma, B by averaging over flux_bias."""
    (
        A_fit,
        f0_fit,
        gamma_fit,
        B_fit,
        A_fit_err,
        f0_fit_err,
        gamma_fit_err,
        B_fit_err,
    ) = xr.apply_ufunc(
        fit_lorentzian,
        da.detuning,
        da,
        input_core_dims=[["detuning"], ["detuning"]],
        output_core_dims=[[]] * 8,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float] * 8,
    )

    total_err = (
        (A_fit_err / A_fit) ** 2
        + (B_fit_err / B_fit) ** 2
        + (gamma_fit_err / gamma_fit) ** 2
        + (f0_fit_err / f0_fit) ** 2
    )
    weights = abs(da.std(dim="detuning") - da.std(dim="detuning").min()) * (1 / total_err)

    A_weighted_avg = (A_fit * weights).sum() / weights.sum()
    gamma_weighted_avg = (gamma_fit * weights).sum() / weights.sum()
    B_weighted_avg = (B_fit * weights).sum() / weights.sum()

    return A_weighted_avg.data, gamma_weighted_avg.data, B_weighted_avg.data


def fit_lorentzian_for_each_detuning_fixed(da: xr.DataArray):
    """
    Perform two-stage fitting:
    1. Fit averaged data to estimate A, gamma, B.
    2. Fit each cut with only f0 free.
    """
    A, gamma, B = estimate_global_parameters(da)

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

    detuning_range = float(da.detuning.max().values - da.detuning.min().values)
    contrast = da.std(dim="detuning")
    min_contrast = max(0.05 * float(contrast.max().values), 1e-6)

    mask = (~peak_freq.isnull()) & (~peak_freq_err.isnull())

    detuning_min = float(da.detuning.min().values)
    detuning_max = float(da.detuning.max().values)
    mask &= (detuning_min < peak_freq) & (peak_freq < detuning_max)
    mask &= peak_freq_err < detuning_range / 2
    mask &= contrast > min_contrast

    peak_freq = peak_freq.where(mask, drop=True)
    peak_freq_err = peak_freq_err.where(mask, drop=True)
    flux_bias = peak_freq.flux_bias.where(mask, drop=True)

    return peak_freq, peak_freq_err, flux_bias


def _slope_stderr(x_in: np.ndarray, y_in: np.ndarray, slope: float, intercept: float) -> float:
    n = len(x_in)
    if n < 2:
        return np.nan
    residuals = y_in - (slope * x_in + intercept)
    dof = max(n - 2, 1)
    s_sq = np.sum(residuals**2) / dof
    x_var = np.sum((x_in - x_in.mean()) ** 2)
    if x_var == 0:
        return np.nan
    return float(np.sqrt(s_sq / x_var))


def _wls_slope_stderr(x_in: np.ndarray, y_err_in: np.ndarray) -> float:
    """Analytical WLS slope uncertainty from inlier peak errors."""
    n = len(x_in)
    if n < 2:
        return np.nan
    weights = 1.0 / np.maximum(y_err_in**2, np.finfo(float).tiny)
    design = np.column_stack([x_in, np.ones(n)])
    try:
        cov = np.linalg.inv(design.T @ (weights[:, None] * design))
    except np.linalg.LinAlgError:
        return np.nan
    return float(np.sqrt(cov[0, 0]))


def fit_linear(x_data, y_data, y_errors=None):
    """
    Fit data using RANSAC linear regression.

    When ``y_errors`` are provided, points are weighted by 1/sigma^2 during
    RANSAC and the slope uncertainty is taken from the WLS covariance on inliers.
    """
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if y_errors is not None:
        mask &= np.isfinite(y_errors) & (y_errors > 0)

    X = np.asarray(x_data)[mask].reshape(-1, 1)
    y = np.asarray(y_data)[mask]
    sample_weight = None
    y_err = None
    if y_errors is not None:
        y_err = np.asarray(y_errors)[mask]
        sample_weight = 1.0 / y_err**2

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y, sample_weight=sample_weight)

    inlier_mask = ransac.inlier_mask_
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    x_in = X[inlier_mask].ravel()
    y_in = y[inlier_mask]
    if y_err is not None:
        slope_err = _wls_slope_stderr(x_in, y_err[inlier_mask])
    else:
        slope_err = _slope_stderr(x_in, y_in, slope, intercept)

    return slope, intercept, inlier_mask, slope_err


def calculate_crosstalk(
    slope: float,
    slope_err: float,
    denominator: float,
    denominator_err: float = 0.0,
) -> tuple[float, float]:
    """Crosstalk alpha = slope / denominator and propagated uncertainty."""
    if not np.isfinite(slope) or not np.isfinite(denominator) or denominator == 0:
        return np.nan, np.nan

    coefficient = slope / denominator
    if not np.isfinite(slope_err):
        return coefficient, np.nan

    rel_cross = (slope_err / slope) ** 2 if slope != 0 else np.inf
    if np.isfinite(denominator_err) and denominator_err > 0:
        rel_den = (denominator_err / denominator) ** 2
    else:
        rel_den = 0.0

    uncertainty = abs(coefficient) * np.sqrt(rel_cross + rel_den)
    return coefficient, uncertainty
