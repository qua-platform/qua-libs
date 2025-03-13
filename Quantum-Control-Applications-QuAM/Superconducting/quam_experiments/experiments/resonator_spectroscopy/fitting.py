import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from qualibrate import QualibrationNode
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from quam_libs.qua_datasets import convert_IQ_to_V, add_amplitude_and_phase
from scipy.optimize import curve_fit


# todo: DOCSTRINGS!
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


def analyze_raw_data(ds: xr.Dataset, node: QualibrationNode):
    # Add the full frequency axis to the dataset
    full_freq = np.array([ds.detuning + node.machine.qubits[q].resonator.RF_frequency for q in node.machine.qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Convert I/Q data into Volts
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add R=sqrt(I**2 + Q**2) and the phase
    ds = add_amplitude_and_phase(ds, subtract_slope_flag=True)
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

    # Fit the resonator line
    fit_results = _fit_lorentzian(ds.R, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results.to_dataset(name="ds_fit"), node)
    return fit_data, fit_results


# Define Lorentzian function
def lorentzian(x, amplitude, center, width, offset):
    return offset - amplitude * width**2 / (width**2 + (x - center) ** 2)


def _fit_lorentzian(da, dim):
    """Perform the fitting process based on the state discrimination flag."""

    freq_guess = da[dim][da.argmin(dim=dim)]  # Frequency at minimum transmission
    amp_guess = da.max(dim=dim) - da.min(dim=dim)  # Peak-to-peak amplitude
    offset_guess = da.mean(dim=dim)  # Average transmission level
    half_max = (da.max(dim=dim) + da.min(dim=dim)) / 2
    above_half = da > half_max
    width_guess = np.abs(da[dim].where(above_half).max(dim=dim) - da[dim].where(above_half).min(dim=dim)) / 2

    def apply_fit(x, y, amplitude, center, width, offset):
        try:
            fit, residuals = curve_fit(lorentzian, x, y, p0=[amplitude, center, width, offset])
            return np.array(fit.tolist() + np.array(residuals).flatten().tolist())
        except RuntimeError as e:
            print("Fit failed:")
            print(f"{amplitude=}, {center=}, {width=}, {offset=}")
            plt.plot(x, lorentzian(x, amplitude, center, width, offset))
            plt.plot(x, y)
            plt.show()

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess.values,
        width_guess,
        offset_guess,
        input_core_dims=[[dim], [dim], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(
        fit_vals=(
            "fit_vals",
            [
                "a",
                "center_freq",
                "width",
                "offset",
                "a_a",
                "a_center_freq",
                "a_width",
                "a_offset",
                "center_freq_a",
                "center_freq_center_freq",
                "center_freq_width",
                "center_freq_offset",
                "width_a",
                "width_center_freq",
                "width_width",
                "width_offset",
                "offset_a",
                "offset_freq",
                "offset_width",
                "offset_offset",
            ],
        )
    )


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array([node.machine.qubits[q].resonator.RF_frequency for q in node.machine.qubits])
    res_freq = fit.sel(fit_vals="center_freq") + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.ds_fit.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the error on the resonator frequency
    res_freq_error = np.sqrt(fit.sel(fit_vals="center_freq_center_freq"))
    fit = fit.assign_coords(res_freq_error=("qubit", res_freq_error.ds_fit.data))
    fit.res_freq_error.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    full_freq = np.array([node.machine.qubits[q].resonator.RF_frequency for q in node.machine.qubits])
    fwhm = np.abs(fit.sel(fit_vals="width") * 2)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.ds_fit.data))
    fit.fwhm.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the error on the FWHM
    fwhm_error = np.sqrt(fit.sel(fit_vals="width_width"))
    fit = fit.assign_coords(fwhm_error=("qubit", fwhm_error.ds_fit.data))
    fit.fwhm_error.attrs = {"long_name": "FWHM", "units": "Hz"}
    # Assess whether the fit was successful or not
    # freq_error_success = res_freq_error.ds_fit.data / res_freq.ds_fit.data < 1
    freq_success = np.abs(res_freq.ds_fit.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.ds_fit.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    # fwhm_error_success = fwhm_error.ds_fit.data / fwhm.ds_fit.data < 1
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
