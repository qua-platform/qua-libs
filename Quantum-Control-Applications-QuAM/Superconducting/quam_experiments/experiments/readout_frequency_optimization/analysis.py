import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant readout frequency optimization experiment fit parameters for a single qubit"""

    optimal_frequency: float
    optimal_detuning: float
    chi: float
    success: bool
    qubit_name: Optional[str] = ""


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tOptimal readout frequency: {1e-9 * fit_results[q]['optimal_frequency']:.3f} GHz (shifted by {1e-6 * fit_results[q]['optimal_detuning']:.2f} MHz) | "
        s_chi = f"chi: {1e-6 * fit_results[q]['chi']:.2f} MHz\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        logger.info(s_qubit + s_freq + s_chi)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], ["I_g", "Q_g", "I_e", "Q_e"])
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g> and |e> as well as the distance between the two blobs D
    ds = ds.assign(
        {
            "D": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
            "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
            "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
        }
    )
    # Add the absolute frequency to the dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "Readout RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds

    # Get the readout detuning as the index of the maximum of the cumulative average of D
    ds_fit["optimal_index"] = ds_fit.D.rolling({"detuning": 5}).mean("detuning").argmax("detuning")
    ds_fit["optimal_detuning"] = ds_fit.detuning.isel(detuning=ds_fit["optimal_index"])
    ds_fit["optimal_frequency"] = ds_fit.full_freq.isel(detuning=ds_fit["optimal_index"])
    # Get the dispersive shift as the distance between the resonator frequency when the qubit is in |g> and |e>
    ds_fit["chi"] = (ds_fit.IQ_abs_e.idxmin(dim="detuning") - ds_fit.IQ_abs_g.idxmin(dim="detuning")) / 2

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {
        q: FitParameters(
            optimal_frequency=float(ds_fit["optimal_frequency"].sel(qubit=q).data),
            optimal_detuning=float(ds_fit["optimal_detuning"].sel(qubit=q).data),
            chi=float(ds_fit["chi"].sel(qubit=q).data),
            success=False,
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results
