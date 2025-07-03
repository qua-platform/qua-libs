from typing import Optional

import numpy as np
import xarray as xr
from qualibration_libs.analysis import lorentzian_dip
from qualibration_libs.analysis.models import S21Resonator


def prepare_resonator_spectroscopy_data(
    ds_raw: xr.Dataset, ds_fit: Optional[xr.Dataset] = None
) -> (xr.Dataset, Optional[xr.Dataset]):
    """
    Prepares resonator spectroscopy datasets for plotting.

    This function enriches the raw and fit datasets with additional, plot-ready
    fields. It uses S21 model fits if they are available, otherwise it falls back to a simple Lorentzian model.
    """
    # --- Raw Data Preparation ---
    ds_raw_processed = ds_raw.copy()
    if "full_freq" in ds_raw_processed:
        ds_raw_processed["full_freq_ghz"] = ds_raw_processed.full_freq / 1e9
    if "detuning" in ds_raw_processed.dims:
        ds_raw_processed.coords["detuning_mhz"] = ("detuning", ds_raw_processed.detuning.values / 1e6)
    if "IQ_abs" in ds_raw_processed:
        ds_raw_processed["iq_abs_mv"] = ds_raw_processed.IQ_abs * 1e3
    if "phase" in ds_raw_processed:
        ds_raw_processed["phase"] = ds_raw_processed.phase
    
    # --- Fit Data Preparation ---
    ds_fit_processed = None
    if ds_fit is not None:
        ds_fit_processed = ds_fit.copy()
        s21_models = ds_fit.attrs.get("s21_models", {})

        # --- Logic to generate curves from S21 models ---
        if s21_models:
            # print("S21 models found. Generating fit curves for plotting.")
            
            # Create empty DataArrays to hold the new fit data
            all_mag_curves = xr.DataArray(np.nan, coords=ds_raw.IQ_abs.coords, dims=ds_raw.IQ_abs.dims)
            all_phase_curves = xr.DataArray(np.nan, coords=ds_raw.phase.coords, dims=ds_raw.phase.dims)

            for qubit_id, fitter in s21_models.items():
                # Check if fit was successful and model exists
                fit_q = ds_fit_processed.sel(qubit=qubit_id)
                if (hasattr(fit_q, 'outcome') and fit_q.outcome.values == "successful" and 
                    fitter.full_s21_model is not None):
                    # Extract magnitude and phase from the fitter's full_s21_model
                    mag_curve = np.abs(fitter.full_s21_model)
                    phase_curve = np.unwrap(np.angle(fitter.full_s21_model))
                    
                    # Flatten the phase for plotting, just like in our standalone plotter
                    if fitter.fit_params and 'cable_delay_s' in fitter.fit_params:
                         delay = fitter.fit_params['cable_delay_s']
                         slope = (-2 * np.pi * delay) * (fitter.frequencies - np.mean(fitter.frequencies))
                         phase_curve -= slope

                    all_mag_curves.loc[dict(qubit=qubit_id)] = mag_curve
                    all_phase_curves.loc[dict(qubit=qubit_id)] = phase_curve
            
            # Assign to the dataset with names matching the plot configs
            ds_fit_processed["fitted_curve"] = all_mag_curves
            ds_fit_processed["fitted_phase_rad"] = all_phase_curves # For the phase plot config
            ds_fit_processed["fitted_curve_mv"] = ds_fit_processed["fitted_curve"] * 1e3
        
        # --- FALLBACK: Original Lorentzian logic ---
        elif "fitted_curve" not in ds_fit_processed and all(p in ds_fit for p in ["amplitude", "position", "width", "base_line"]):
            print("S21 models not found. Falling back to Lorentzian curve generation.")
            # Calculate fitted curve from parameters
            required_params = ["amplitude", "position", "width", "base_line", "outcome"]
            if all(p in ds_fit_processed for p in required_params):
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