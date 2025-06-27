from typing import Optional

import numpy as np
import xarray as xr
from qualibration_libs.analysis import lorentzian_dip


def prepare_resonator_spectroscopy_data(
    ds_raw: xr.Dataset, ds_fit: Optional[xr.Dataset] = None
) -> (xr.Dataset, Optional[xr.Dataset]):
    """
    Prepares resonator spectroscopy datasets for plotting.

    This function enriches the raw and fit datasets with additional, plot-ready
    fields like unit-converted values and calculated fitted curves.

    Args:
        ds_raw: The raw dataset from the experiment.
        ds_fit: The dataset containing fit parameters.

    Returns:
        A tuple of (ds_raw_processed, ds_fit_processed).
    """
    ds_raw_processed = ds_raw.copy()
    if "full_freq" in ds_raw_processed:
        ds_raw_processed["full_freq_ghz"] = ds_raw_processed.full_freq / 1e9
    if "detuning" in ds_raw_processed.dims:
        ds_raw_processed.coords["detuning_mhz"] = ("detuning", ds_raw_processed.detuning.values / 1e6)
    if "IQ_abs" in ds_raw_processed:
        ds_raw_processed["iq_abs_mv"] = ds_raw_processed.IQ_abs * 1e3
    if "phase" in ds_raw_processed:
        ds_raw_processed["phase"] = ds_raw_processed.phase

    ds_fit_processed = None
    if ds_fit is not None:
        ds_fit_processed = ds_fit.copy()
        
        # Calculate fitted curve from parameters
        required_params = ["amplitude", "position", "width", "base_line", "outcome"]
        if "fitted_curve" not in ds_fit_processed and all(p in ds_fit_processed for p in required_params):
            all_curves = xr.DataArray(
                np.nan,
                coords=[ds_fit_processed.qubit, ds_raw.detuning],
                dims=["qubit", "detuning"]
            )
            for qubit_id in ds_fit_processed.qubit.values:
                fit_q = ds_fit_processed.sel(qubit=qubit_id)
                if fit_q.outcome.values == "successful":
                    curve = lorentzian_dip(
                        ds_raw.detuning.values,
                        float(fit_q.amplitude.values),
                        float(fit_q.position.values),
                        float(fit_q.width.values) / 2,
                        float(fit_q.base_line.mean().values),
                    )
                    all_curves.loc[dict(qubit=qubit_id)] = curve
            ds_fit_processed["fitted_curve"] = all_curves

        # Add other plot-ready fields to the fit dataset
        if "full_freq" not in ds_fit_processed and "full_freq" in ds_raw_processed:
            ds_fit_processed["full_freq"] = ds_raw_processed.full_freq
        if "full_freq" in ds_fit_processed:
            ds_fit_processed["full_freq_ghz"] = ds_fit_processed.full_freq / 1e9
        if "fitted_curve" in ds_fit_processed:
            ds_fit_processed["fitted_curve_mv"] = ds_fit_processed.fitted_curve * 1e3

    return ds_raw_processed, ds_fit_processed 