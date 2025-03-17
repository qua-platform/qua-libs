import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from quam_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips


@dataclass
class ResonatorFit:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    success: bool
    qubit_name: Optional[str] = ""


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
        Expected variables: 'frequency', 'frequency_error', 'fwhm', 'fwhm_error', 'success'.
        Expected coordinate: 'qubit'.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    Returns:
    --------
    None

    Example:
    --------
        >>> log_fitted_results(fit_results)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in fit_results.keys():
        s_freq = f"Resonator frequency for qubit {q} : {1e-9 * fit_results[q]['frequency']:.3f} GHz --> "
        # s_freq = f"Resonator frequency for qubit {q} : {1e-3 * fit_results[q]['frequency']:.2f} +/- {1e-3 * fit_results[q]['frequency_error']:.2f} us --> "
        s_fwhm = f"FWHM for qubit {q} : {1e-3 * fit_results[q]['fwhm']:.1f} kHz --> "
        # s_fwhm = f"FWHM for qubit {q} : {1e-3 * fit_results[q]['fwhm']:.2f} +/- {1e-3 * fit_results[q]['fwhm_error']:.2f} us --> "
        if fit_results[q]["success"]:
            logger.info(s_freq + "SUCCESS!")
            logger.info(s_fwhm + "SUCCESS!")
        else:
            logger.error(s_freq + "FAIL!")
            logger.error(s_fwhm + "FAIL!")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_resonators(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, ResonatorFit]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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
    # Add the RF frequency
    # Fit the resonator line
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: ResonatorFit(
            qubit_name=q,
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
