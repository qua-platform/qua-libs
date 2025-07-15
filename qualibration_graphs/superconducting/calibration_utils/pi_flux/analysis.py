import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_decay_exp, fit_oscillation, peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant exponential fit parameters from the pi flux experiment."""

    success: bool
    amplitude: float  # Amplitude of the exponential fit (a parameter)
    offset: float  # Offset/baseline of the exponential fit
    decay_rate: float  # Decay rate of the exponential fit
    amplitude_error: float  # Uncertainty in amplitude (sqrt of a_a from covariance)
    offset_error: float  # Uncertainty in offset (sqrt of offset_offset from covariance)
    decay_rate_error: float  # Uncertainty in decay rate (sqrt of decay_decay from covariance)
    sweet_spot_freq: float  # Calculated sweet spot frequency (offset + amplitude)
    frequency_shift: float  # Calculated frequency shift (amplitude)


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the exponential fit results for all qubits

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
        fit_param = fit_results[q]
        s_qubit = f"Results for qubit {q}: "

        if fit_param.success:
            s_qubit += " SUCCESS!\n"
            s_amplitude = f"\tAmplitude: {fit_param.amplitude:.3e} ± {fit_param.amplitude_error:.3e} Hz\n"
            s_offset = f"\tOffset: {fit_param.offset:.3e} ± {fit_param.offset_error:.3e} Hz\n"
            s_decay = f"\tDecay rate: {fit_param.decay_rate:.3e} ± {fit_param.decay_rate_error:.3e}\n"
            s_sweet_spot = f"\tSweet spot frequency: {fit_param.sweet_spot_freq:.3e} Hz\n"
            s_shift = f"\tFrequency shift: {fit_param.frequency_shift:.3e} Hz\n"
            log_callable(s_qubit + s_amplitude + s_offset + s_decay + s_sweet_spot + s_shift)
        else:
            s_qubit += " FAIL!\n"
            log_callable(s_qubit)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V,
    or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
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

    peak_freq = peaks_dips(ds.I, dim="detuning", prominence_factor=5)
    fit_results_ds = xr.merge([ds, peak_freq.position.rename("peak_freq")])


    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_ds, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""

    # Fit an exponential function to the peak_freq data
    # We need to determine the appropriate dimension to fit against
    # Based on the context of pi_flux experiments, this could be flux bias, phase, or similar

    # First, let's check what dimensions are available in peak_freq
    # Common dimensions in flux experiments include 'flux_bias', 'phase', or similar sweep parameters
    peak_freq_dims = list(fit.peak_freq.dims)

    # Remove 'qubit' dimension as we want to fit each qubit separately
    fit_dims = [dim for dim in peak_freq_dims if dim != "qubit"]

    if not fit_dims:
        raise ValueError("No suitable dimension found for fitting peak_freq")

    # Use the first non-qubit dimension for fitting (typically the sweep parameter)
    fit_dim = fit_dims[0]

    # Fit exponential decay to peak_freq along the identified dimension for each qubit
    try:
        # Initialize lists to store fit results for all qubits
        all_fit_results = []
        qubit_names = []

        # Perform fit for each qubit individually
        for q in fit.qubit.values:
            try:
                # Extract peak_freq data for this specific qubit
                qubit_peak_freq = fit.peak_freq.sel(qubit=q)

                # Perform exponential fit for this qubit
                qubit_fit = fit_decay_exp(qubit_peak_freq, fit_dim)
                all_fit_results.append(qubit_fit)
                qubit_names.append(q)

            except Exception as qubit_error:
                print(f"Fit failed for qubit {q}: {qubit_error}")
                # Create NaN results for failed qubit
                nan_values = np.full(12, np.nan)  # 12 fit_vals from fit_decay_exp
                qubit_fit = xr.DataArray(
                    nan_values,
                    dims=["fit_vals"],
                    coords={
                        "fit_vals": [
                            "a",
                            "offset",
                            "decay",
                            "a_a",
                            "a_offset",
                            "a_decay",
                            "offset_a",
                            "offset_offset",
                            "offset_decay",
                            "decay_a",
                            "decay_offset",
                            "decay_decay",
                        ]
                    },
                )
                all_fit_results.append(qubit_fit)
                qubit_names.append(q)

        # Combine all qubit results into a single DataArray
        if all_fit_results:
            exponential_fit = xr.concat(all_fit_results, dim="qubit")
            exponential_fit = exponential_fit.assign_coords(qubit=qubit_names)
            fit = fit.assign({"exp_fit": exponential_fit})

            # Add metadata to the exponential fit results
            amplitude = fit.exp_fit.sel(fit_vals="a")
            amplitude.attrs = {"long_name": "amplitude", "units": "Hz"}

            offset = fit.exp_fit.sel(fit_vals="offset")
            offset.attrs = {"long_name": "offset", "units": "Hz"}

            decay_rate = fit.exp_fit.sel(fit_vals="decay")
            decay_rate.attrs = {"long_name": "decay rate", "units": f"1/{fit_dim}"}

            # Calculate success criteria based on fit quality for each qubit
            # Check if any of the fit parameters are NaN
            nan_amplitude = np.isnan(amplitude)
            nan_offset = np.isnan(offset)
            nan_decay = np.isnan(decay_rate)
            nan_success = nan_amplitude | nan_offset | nan_decay

            success_criteria = ~nan_success
            fit = fit.assign({"success": success_criteria})

            # Create fit results dictionary
            fit_results = {}
            for q in fit.qubit.values:
                # Extract fitted parameters for each qubit
                qubit_amplitude = float(amplitude.sel(qubit=q).values)
                qubit_offset = float(offset.sel(qubit=q).values)
                qubit_decay = float(decay_rate.sel(qubit=q).values)
                qubit_success = bool(success_criteria.sel(qubit=q).values)

                # Extract error estimates from the covariance matrix diagonal elements
                amplitude_variance = float(fit.exp_fit.sel(fit_vals="a_a", qubit=q).values)
                offset_variance = float(fit.exp_fit.sel(fit_vals="offset_offset", qubit=q).values)
                decay_variance = float(fit.exp_fit.sel(fit_vals="decay_decay", qubit=q).values)

                # Calculate standard errors (square root of variances)
                amplitude_error = np.sqrt(np.abs(amplitude_variance)) if not np.isnan(amplitude_variance) else np.nan
                offset_error = np.sqrt(np.abs(offset_variance)) if not np.isnan(offset_variance) else np.nan
                decay_error = np.sqrt(np.abs(decay_variance)) if not np.isnan(decay_variance) else np.nan

                # Calculate derived parameters
                sweet_spot_freq = qubit_offset + qubit_amplitude
                frequency_shift = qubit_amplitude

                fit_results[q] = FitParameters(
                    success=qubit_success,
                    amplitude=qubit_amplitude,
                    offset=qubit_offset,
                    decay_rate=qubit_decay,
                    amplitude_error=amplitude_error,
                    offset_error=offset_error,
                    decay_rate_error=decay_error,
                    sweet_spot_freq=sweet_spot_freq,
                    frequency_shift=frequency_shift,
                )
        else:
            raise ValueError("No successful fits for any qubit")

    except Exception as e:
        # If fitting fails, create default/failed results
        print(f"Exponential fitting failed: {e}")
        fit_results = {}
        for q in fit.qubit.values:
            fit_results[q] = FitParameters(
                success=False,
                amplitude=np.nan,
                offset=np.nan,
                decay_rate=np.nan,
                amplitude_error=np.nan,
                offset_error=np.nan,
                decay_rate_error=np.nan,
                sweet_spot_freq=np.nan,
                frequency_shift=np.nan,
            )

        # Add a failed fit indicator to the dataset
        success_array = xr.DataArray([False] * len(fit.qubit), dims=["qubit"], coords={"qubit": fit.qubit})
        fit = fit.assign({"success": success_array})

    return fit, fit_results
