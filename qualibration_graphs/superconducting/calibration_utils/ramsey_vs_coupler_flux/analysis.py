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
    with contextlib.redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"):
        fit_data = fit_oscillation_decay_exp(ds.state, "idle_times")
    fit_data.attrs = {"long_name": "time", "units": "µs"}

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

    measured_qubit_role = node.parameters.measured_qubit
    qubit_pair_names = ds.qubit_pair.values
    rf_freqs = []
    for qp_name in qubit_pair_names:
        qp = node.machine.qubit_pairs[str(qp_name)]
        qubit = qp.qubit_control if measured_qubit_role == "control" else qp.qubit_target
        rf_freqs.append(qubit.xy.RF_frequency)
    rf_freq_da = xr.DataArray(rf_freqs, dims=["qubit_pair"], coords={"qubit_pair": qubit_pair_names})

    ds_fit["qubit_frequency"] = rf_freq_da - frequency * 1e9 + node.parameters.frequency_detuning_in_mhz * 1e6
    ds_fit["qubit_frequency"].attrs = {
        "long_name": "qubit frequency",
        "units": "Hz",
        "measured_qubit": measured_qubit_role,
    }

    return ds_fit
