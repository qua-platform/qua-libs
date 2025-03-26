import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from quam_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

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
    # for q in fit_results.keys():
    #     s_qubit = f"Results for qubit {q}: "
    #     s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
    #     s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
    #     s_angle = (
    #         f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
    #     )
    #     s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
    #     s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
    #     if fit_results[q]["success"]:
    #         s_qubit += " SUCCESS!\n"
    #     else:
    #         s_qubit += " FAIL!\n"
    #     logger.info(
    #         s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180
    #     )
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # full_freq = np.array(
    #     [ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]]
    # )
    # ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    # ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    pass
    return ds

# @dataclass
# class RamseyFit:
#     """Stores the relevant Ramsey experiment fit parameters for a single qubit"""
#
#     freq_offset: float
#     decay: float
#     decay_error: float
#
#     qubit_name: Optional[str] = ""
#     raw_fit_results: Optional[xr.Dataset] = None
#
#     def log_frequency_offset(self, logger=None):
#         if logger is None:
#             logger = logging.getLogger(__name__)
#         logger.info(
#             f"Frequency offset for qubit {self.qubit_name} : {self.freq_offset / 1e6:.2f} MHz "
#         )
#
#     def log_t2(self, logger=None):
#         if logger is None:
#             logger = logging.getLogger(__name__)
#         logger.info(f"T2* for qubit {self.qubit_name} : {1e6 * self.decay:.2f} us")

# def fit_frequency_detuning_and_t2_decay(
#     ds: xr.Dataset, node_parameters: Parameters
# ) -> dict[str, RamseyFit]:
#     """
#     Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.
#
#     Returns:
#         dict: Dictionary containing fit results.
#     """
#     fit = fit_ramsey_oscillations_with_exponential_decay(
#         ds, node_parameters.use_state_discrimination
#     )
#
#     frequency, decay, tau, tau_error = extract_relevant_fit_parameters(fit)
#
#     detuning = int(node_parameters.frequency_detuning_in_mhz * 1e6)
#
#     freq_offset, decay, decay_error = calculate_fit_results(
#         frequency, tau, tau_error, fit, detuning
#     )
#
#     fits = {
#         q.name: RamseyFit(
#             qubit_name=q.item(),
#             freq_offset=1e9 * freq_offset.loc[q].values,
#             decay=decay.loc[q].values,
#             decay_error=decay_error.loc[q].values,
#             raw_fit_results=fit.to_dataset(name="fit"),
#         )
#         for q in freq_offset.qubit
#     }
#
#     return fits

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
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
    # from quam_experiments.analysis.fit import fit_oscillation_decay_exp
    # if use_state_discrimination:
    #     fit = fit_oscillation_decay_exp(ds.state, "time")
    # else:
    #     fit = fit_oscillation_decay_exp(ds.I, "time")
    # return fit

    ds_fit = ds
    fit_results = {
        q: FitParameters(
            success=False,
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
    # # Add metadata to fit results
    # fit.attrs = {"long_name": "time", "units": "µs"}
    #
    # # Add calculated metadata to the dataset
    # frequency = fit.sel(fit_vals="f")
    # frequency.attrs = {"long_name": "frequency", "units": "MHz"}
    # frequency = frequency.where(frequency > 0, drop=True)
    #
    # decay = fit.sel(fit_vals="decay")
    # decay.attrs = {"long_name": "decay", "units": "nSec"}
    #
    # decay_res = fit.sel(fit_vals="decay_decay")
    # decay_res.attrs = {"long_name": "decay residual", "units": "nSec"}
    #
    # tau = 1 / decay
    # tau.attrs = {"long_name": "T2*", "units": "uSec"}
    #
    # tau_error = tau * (np.sqrt(decay_res) / decay)
    # tau_error.attrs = {"long_name": "T2* error", "units": "uSec"}

# def calculate_fit_results(frequency, tau, tau_error, fit, detuning):
#     """
#     Calculate fit results such as frequency offset, decay, and decay error.
#
#     Parameters:
#         frequency (xarray.DataArray): Frequency data.
#         tau (xarray.DataArray): Tau values.
#         tau_error (xarray.DataArray): Tau error values.
#         fit (xarray.DataArray): Fit results.
#         detuning (float): Detuning parameter in Hz.
#
#     Returns:
#         tuple: Frequency offset, decay, and decay error.
#     """
#     within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="sign") == 1
#     positive_shift = frequency.sel(sign=1) > frequency.sel(sign=-1)
#     freq_offset = (
#         within_detuning * (frequency * fit.sign).mean(dim="sign")
#         + ~within_detuning * positive_shift * frequency.mean(dim="sign")
#         - ~within_detuning * ~positive_shift * frequency.mean(dim="sign")
#     )
#
#     decay = 1e-9 * tau.mean(dim="sign")
#     decay_error = 1e-9 * tau_error.mean(dim="sign")
#
#     return freq_offset, decay, decay_error