"""Data analysis for Ramsey versus coupler flux calibration.

Extracts oscillation frequencies from the Ramsey data.  No model
(e.g. parabola) is fitted to the resulting frequency-vs-coupler-flux curve.
"""

import io
import contextlib
from unittest.mock import patch

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation_decay_exp, oscillation_decay_exp


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Process raw dataset (placeholder)."""
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Extract oscillation frequency per coupler-flux slice.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``state`` variable, dims (qubit_pair, coupler_flux, idle_times).
    node : QualibrationNode
        Node instance (used to read parameters and qubit pair info).

    Returns
    -------
    xr.Dataset
        Dataset augmented with per-slice fit results, the fitted state curves,
        and the absolute qubit frequency ``qubit_frequency`` (Hz).
    """
    # Fit a * exp(decay * t) * cos(2π f t + phi) + offset to each (qubit_pair, coupler_flux)
    # slice. The context managers suppress the debug print and popup plot that the library
    # emits when curve_fit fails to converge on a noisy/flat slice.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.plot"),
    ):
        fit_data = fit_oscillation_decay_exp(ds.state, "idle_times")
    fit_data.attrs = {"long_name": "time", "units": "µs"}

    # Evaluate the fitted model on the original time axis for plotting
    fitted = oscillation_decay_exp(
        ds.state.idle_times,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )

    frequency = np.abs(fit_data.sel(fit_vals="f"))

    ds_fit = ds.merge(fit_data.rename("fit_results"))
    ds_fit["fitted_state"] = fitted
    ds_fit["artificial_detuning"] = node.parameters.frequency_detuning_in_mhz
    ds_fit["frequency"] = frequency

    # Convert Ramsey detuning frequency back to absolute qubit frequency:
    # f_qubit = RF_freq - f_Ramsey + f_detuning
    qubit_names = ds.qubit.values
    rf_freqs = [q.xy.RF_frequency for q in node.namespace["measured_qubits"]]
    rf_freq_da = xr.DataArray(rf_freqs, dims=["qubit"], coords={"qubit": qubit_names})

    ds_fit["qubit_frequency"] = rf_freq_da - frequency * 1e9 + node.parameters.frequency_detuning_in_mhz * 1e6
    ds_fit["qubit_frequency"].attrs = {
        "long_name": "qubit frequency",
        "units": "Hz",
    }

    return ds_fit
