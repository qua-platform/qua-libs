"""Analysis utilities for XY-Z delay calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

__all__ = [
    "FitParameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
]


@dataclass
class FitParameters:
    """Stores XY-Z delay fit parameters: success status and extracted flux delay."""

    success: bool
    flux_delay: int


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        success = res["success"] if isinstance(res, dict) else res.success
        flux_delay = res["flux_delay"] if isinstance(res, dict) else res.flux_delay
        status = "SUCCESS" if success else "FAIL"
        log_callable(f"Qubit {q}: {status} | flux_delay = {flux_delay} ns")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    data = "state" if hasattr(ds, "state") else "I"

    difference = ds[data].sel(init_state="e") - ds[data].sel(init_state="g")

    if data == "I":
        difference -= difference.mean()

    ds["difference"] = difference
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the XY-Z delay response for each qubit and extract flux-delay parameters.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing per-qubit `difference` traces versus `relative_time`.
    node : QualibrationNode
        Node context used by the fitter (e.g., measured qubit objects and config).

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        `dfit`: dataset with fitted curve (`fit`) and extracted values
        (`flux_delay`, `flux_delay_std`, `success`) per qubit.
        `fit_results`: summarized fit outcomes keyed by qubit name.
    """

    dfit = ds.groupby("qubit").apply(fit_routine, node=node)

    fit_results = _extract_relevant_fit_parameters(dfit, node)

    return dfit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for q in fit.qubit.data:
        fit_results[q] = FitParameters(
            success=fit.success.sel(qubit=q).data, flux_delay=fit.flux_delay.sel(qubit=q).data
        )
    return fit_results


def triangle_peak(t, amp, t0, half_width, offset):
    """
    Cross-correlation of two equal-duration rectangular pulses.
    Both XY and coupler flux pulses have the same length by construction
    (flux waveform is passed in already matching x180 length).
    """
    return amp * np.maximum(0, 1 - np.abs(t - t0) / half_width) + offset


def fit_delay_trace(da, xy_duration: float):
    """Fit a single delay scan trace to a triangle peak and attach fit metadata."""
    x = da.relative_time.data
    y = da.difference.data[0]

    # Smooth before argmax to prevent a single noise spike from seeding the fit
    y_smooth = uniform_filter1d(y, size=5)
    t0_guess = float(x[np.argmax(y_smooth)])
    # 10th percentile as a proxy for the signal floor (robust to the peak inflating the mean)
    amp_guess = float(y_smooth.max()) - float(np.percentile(y, 10))
    # Same floor as the peak (could be noise or baseline offset).
    offset_guess = float(np.percentile(y, 10))

    # Only consider samples far from the peak (coupler fully misaligned)
    wing_mask = np.abs(x - t0_guess) > xy_duration
    # Noise from wings (far from peak, coupler fully misaligned); fall back to
    # full-signal std if too few wing points (peak near scan edge).
    noise_std = float(np.std(y[wing_mask])) if wing_mask.sum() > 10 else float(np.std(y))

    try:
        p0 = [amp_guess, t0_guess, xy_duration, offset_guess]
        bounds = (
            [0, x.min(), 0.5 * xy_duration, -np.inf],
            [np.inf, x.max(), 1.5 * xy_duration, np.inf],
        )
        popt, pcov = curve_fit(triangle_peak, x, y, p0=p0, bounds=bounds, maxfev=3000)
        amp, flux_delay, half_width, offset = popt
        # Uncertainty on t0 from the diagonal of the covariance matrix
        flux_delay_std = np.sqrt(pcov[1, 1])

        # Ensure the peak is far enough from the scan edges (prevent edge effects)
        assert (
            x.min() + xy_duration < flux_delay < x.max() - xy_duration
        ), f"Peak at {flux_delay:.2f} ns is too close to scan edge — increase zeros_each_side."
        # Ensure the uncertainty is small enough (prevent overfitting)
        assert flux_delay_std < 5.0, f"Fit uncertainty too large: {flux_delay_std:.2f} ns."
        # Ensure the amplitude is positive (prevent negative signals)
        assert amp > 0, "Fitted amplitude is negative — check signal polarity."
        # Ensure the peak is significant enough (prevent noise-only fits)
        assert amp / noise_std > 3.0, (
            f"Fitted peak not significant: SNR = {amp / noise_std:.1f} " f"(amp={amp:.3f}, noise={noise_std:.3f})."
        )

        y_fit = triangle_peak(x, *popt)
        da = da.assign(fit=xr.DataArray(y_fit, coords={"relative_time": x}))
        da = da.assign(flux_delay=flux_delay, flux_delay_std=flux_delay_std, success=True)

    except Exception as e:
        # Fallback: use argmax of the raw signal (no smoothing)
        y_fit = np.full_like(x, np.nan, dtype=float)
        da = da.assign(fit=xr.DataArray(y_fit, coords={"relative_time": x}))
        # 10th percentile as a proxy for the signal floor (robust to the peak inflating the mean)
        baseline = np.percentile(y, 10)
        # Use argmax of the raw signal (no smoothing)
        flux_delay = float(x[np.argmax(y - baseline)])
        print(f"{da.qubit.data}: fit failed ({e}), argmax fallback → {flux_delay:.2f} ns")
        da = da.assign(flux_delay=flux_delay, flux_delay_std=np.nan, success=False)

    return da


def fit_routine(da, node):
    qubit = next(q for q in node.namespace["qubits"] if q.id == da.qubit.item())
    xy_duration = qubit.xy.operations["x180"].length  # ns
    return fit_delay_trace(da, xy_duration)
