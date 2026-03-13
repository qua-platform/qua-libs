import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy parity-diff fit parameters for a single qubit"""

    frequency: float
    relative_freq: float
    fwhm: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Restructure the per-qubit pdiff variables into a single DataArray with a qubit dimension."""
    qubits = node.namespace["qubits"]
    qubit_names = [q.name for q in qubits]

    # Find pdiff variables from the dataset and pair them with qubits by index
    pdiff_vars = sorted([v for v in ds.data_vars if v.startswith("pdiff_")])

    # Build a combined pdiff DataArray with qubit dimension
    first = ds[pdiff_vars[0]]
    if "qubit" in first.dims:
        # Already has qubit dimension from the data fetcher
        pdiff = first.assign_coords(qubit=qubit_names)
    else:
        # Stack separate 1D per-qubit arrays into a 2D array
        pdiff = xr.DataArray(
            np.array([ds[v].values for v in pdiff_vars]),
            dims=["qubit", "detuning"],
            coords={"qubit": qubit_names, "detuning": ds.detuning},
        )
    ds = ds.assign({"pdiff": pdiff})

    # Add full frequency coordinate
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit Larmor frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The node containing parameters and namespace.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict
        Dictionary of FitParameters per qubit.
    """
    ds_fit = ds
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    fit_vals = peaks_dips(ds_fit.pdiff, dim="detuning", prominence_factor=5)
    ds_fit = xr.merge([ds_fit, fit_vals])
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted qubit frequency
    full_freq = np.array([q.xy.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    rel_freq = fit.position
    fit = fit.assign({"res_freq": ("qubit", res_freq.data)})
    fit = fit.assign({"relative_freq": ("qubit", rel_freq.data)})
    fit.res_freq.attrs = {"long_name": "qubit Larmor frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign({"fwhm": fwhm})
    fit.fwhm.attrs = {"long_name": "qubit fwhm", "units": "Hz"}

    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    success_criteria = freq_success & fwhm_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            relative_freq=fit.sel(qubit=q).relative_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
