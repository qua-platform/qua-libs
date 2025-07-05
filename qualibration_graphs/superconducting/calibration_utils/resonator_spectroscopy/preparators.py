from typing import Any, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .plotting import (plot_raw_amplitude_with_fit, plot_raw_phase,
                       plotly_plot_raw_amplitude_with_fit,
                       plotly_plot_raw_phase)


def create_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[Any],
    ds_fit: Optional[xr.Dataset] = None,
) -> Any:
    """
    Create plotly figure using the original plotting logic to maintain exact visual output.
    """
    config = plot_configs[0]
    if "Phase" in config.layout.title:
        return plotly_plot_raw_phase(ds_raw, qubits)
    elif "Amplitude" in config.layout.title:
        return plotly_plot_raw_amplitude_with_fit(ds_raw, qubits, ds_fit)
    else:
        raise ValueError("Unknown plot config")


def create_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[Any],
    ds_fit: Optional[xr.Dataset] = None,
) -> Any:
    """
    Create matplotlib figure using the original plotting logic to maintain exact visual output.
    """
    config = plot_configs[0]
    if "Phase" in config.layout.title:
        return plot_raw_phase(ds_raw, qubits)
    elif "Amplitude" in config.layout.title:
        return plot_raw_amplitude_with_fit(ds_raw, qubits, ds_fit)
    else:
        raise ValueError("Unknown plot config")


def prepare_resonator_spectroscopy_data(
    ds_raw: xr.Dataset, ds_fit: Optional[xr.Dataset] = None
) -> (xr.Dataset, Optional[xr.Dataset]):
    """
    Prepares resonator spectroscopy datasets for plotting.

    This function enriches the raw and fit datasets with additional, plot-ready
    fields. It uses Lorentzian model fits if they are available.
    """
    # --- Raw Data Preparation ---
    ds_raw_processed = ds_raw.copy()
    if "full_freq" in ds_raw_processed:
        ds_raw_processed["full_freq_GHz"] = ds_raw_processed.full_freq / 1e9
    if "detuning" in ds_raw_processed.dims:
        ds_raw_processed.coords["detuning_MHz"] = ("detuning", ds_raw_processed.detuning.values / 1e6)
    if "IQ_abs" in ds_raw_processed:
        ds_raw_processed["IQ_abs_mV"] = ds_raw_processed.IQ_abs * 1e3
    if "phase" in ds_raw_processed:
        ds_raw_processed["phase"] = ds_raw_processed.phase

    # --- Fit Data Preparation ---
    ds_fit_processed = None
    if ds_fit is not None:
        ds_fit_processed = ds_fit.copy()

        # Only Lorentzian logic, no S21 logic
        if "fitted_curve" not in ds_fit_processed and all(p in ds_fit for p in ["amplitude", "position", "width", "base_line"]):
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
            ds_fit_processed["full_freq_GHz"] = ds_fit_processed.full_freq / 1e9
        if "fitted_curve" in ds_fit_processed:
            ds_fit_processed["fitted_data_mV"] = ds_fit_processed.fitted_curve * 1e3

    return ds_raw_processed, ds_fit_processed 