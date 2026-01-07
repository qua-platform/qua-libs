import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy import optimize, signal


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    avoided_crossing_flux_biases: list[float]
    """List of flux bias values where avoided crossings occur (V)."""
    num_crossings: int
    """Number of avoided crossings found."""
    hyperbolic_fit_params: dict | None = None
    """
    Parameters of the fitted hyperbolic function.
    Keys: 'f0' (center_freq), 'delta_f' (gap), 'g' (coupling), 'phi0' (offset)
    """


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        num_crossings = fit_results[q]["num_crossings"]
        s_crossings = f"Found {num_crossings} avoided crossing(s) | "
        if num_crossings > 0:
            crossings_str = ", ".join([f"{fc * 1e3:.1f} mV" for fc in fit_results[q]["avoided_crossing_flux_biases"]])
            s_crossings += f"at: {crossings_str}"
        else:
            s_crossings += "none found"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_crossings)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset by converting I/Q quadratures to V and adding RF frequency coordinates.
    """

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Find all avoided crossings for each qubit.

    The method:
    1. Finds the peak frequency for each flux bias point
    2. Identifies all avoided crossings (discontinuities) in the frequency vs flux curve

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with dimensions (qubit, detuning, flux_bias).
    node : QualibrationNode
        The calibration node containing parameters.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the fit results and dictionary of fit parameters per qubit.
    """
    ds_fit = ds.copy()

    # Find peak frequency for each flux bias point
    peak_freq = peaks_dips(ds.I, dim="detuning", prominence_factor=5)
    peak_freq_vs_flux = peak_freq.position  # Shape: (qubit, flux_bias)

    # Store the peak frequency curve
    ds_fit["peak_frequency"] = peak_freq_vs_flux

    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node, peak_freq_vs_flux)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, peak_freq_vs_flux: xr.DataArray):
    """
    Extract relevant fit parameters by identifying all avoided crossings.

    Parameters:
    -----------
    fit : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node.
    peak_freq_vs_flux : xr.DataArray
        Peak frequency vs flux bias, shape (qubit, flux_bias).

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Updated fit dataset and dictionary of fit parameters.
    """
    fit_results = {}

    for q in node.namespace["qubits"]:
        q_name = q.name
        try:
            # Get peak frequency vs flux for this qubit
            peak_freq_1d = peak_freq_vs_flux.sel(qubit=q_name)

            # Remove NaN values
            valid_mask = ~np.isnan(peak_freq_1d)
            flux_bias_valid = peak_freq_1d.flux_bias[valid_mask]
            peak_freq_valid = peak_freq_1d[valid_mask]

            if len(peak_freq_valid) < 5:
                # Not enough data points
                fit_results[q_name] = FitParameters(
                    success=False,
                    avoided_crossing_flux_biases=[],
                    num_crossings=0,
                )
                continue

            # Calculate derivative to find discontinuities (avoided crossings)
            # Use a smoothed derivative to reduce noise
            flux_bias_array = flux_bias_valid.data
            peak_freq_array = peak_freq_valid.data

            # Smooth the data slightly to reduce noise
            if len(peak_freq_array) > 5:
                window_size = min(5, len(peak_freq_array) // 3)
                if window_size % 2 == 0:
                    window_size += 1
                peak_freq_smooth = signal.savgol_filter(peak_freq_array, window_size, 2)
            else:
                peak_freq_smooth = peak_freq_array

            # Store smoothed peak frequency for plotting
            # Initialize smoothed_peak_frequency if not exists
            if "smoothed_peak_frequency" not in fit.data_vars:
                qubit_names_all = [q.name for q in node.namespace["qubits"]]
                flux_bias_all = fit.flux_bias.data
                smoothed_init = np.full((len(qubit_names_all), len(flux_bias_all)), np.nan)
                fit["smoothed_peak_frequency"] = (
                    ["qubit", "flux_bias"],
                    smoothed_init,
                    {"long_name": "Smoothed peak frequency", "units": "Hz"},
                )
            # Store smoothed data for this qubit
            qubit_idx = [q.name for q in node.namespace["qubits"]].index(q_name)
            for i, flux_val in enumerate(flux_bias_array):
                flux_idx = np.argmin(np.abs(fit.flux_bias.data - flux_val))
                fit["smoothed_peak_frequency"][qubit_idx, flux_idx] = peak_freq_smooth[i]

            # Calculate derivative
            dfreq_dflux = np.gradient(peak_freq_smooth, flux_bias_array)

            # Find large changes in derivative (avoided crossings)
            # Use a threshold based on the standard deviation of the derivative
            threshold = 3 * np.std(dfreq_dflux)

            # Find clusters of large derivatives (avoided crossings)
            crossing_indices = []
            if np.any(np.abs(dfreq_dflux) > threshold):
                # Find start and end of each crossing region
                large_deriv_mask = np.abs(dfreq_dflux) > threshold
                diff_mask = np.diff(large_deriv_mask.astype(int))
                starts = np.where(diff_mask == 1)[0]
                ends = np.where(diff_mask == -1)[0]

                # Handle edge cases
                if len(starts) > len(ends):
                    ends = np.append(ends, len(large_deriv_mask) - 1)
                if len(ends) > len(starts):
                    starts = np.insert(starts, 0, 0)

                # Find the center of each crossing region
                for start, end in zip(starts, ends):
                    crossing_region = np.arange(start, end + 1)
                    if len(crossing_region) > 0:
                        # Find the point with maximum derivative magnitude in this region
                        max_idx = crossing_region[np.argmax(np.abs(dfreq_dflux[crossing_region]))]
                        crossing_indices.append(max_idx)

            # If no crossings found via clustering, try peak finding
            if len(crossing_indices) == 0:
                min_distance = max(1, len(dfreq_dflux) // 20)  # Minimum distance between peaks
                peak_indices, _ = signal.find_peaks(np.abs(dfreq_dflux), height=threshold, distance=min_distance)
                crossing_indices = sorted(peak_indices.tolist())

            # Get flux bias values at crossings
            crossing_flux_biases = [float(flux_bias_array[idx]) for idx in crossing_indices]
            crossing_flux_biases = sorted(crossing_flux_biases)  # Sort by flux bias value

            # Fit hyperbolic function to the peak frequency vs flux data
            # Hyperbolic form: f(φ) = f₀ ± √((Δf/2)² + (g·(φ-φ₀))²)
            hyperbolic_fit_params = None
            try:
                # Define hyperbolic function for avoided crossing
                def hyperbolic_func(phi, f0, delta_f, g, phi0):
                    """Hyperbolic function for avoided crossing: f(φ) = f₀ + √((Δf/2)² + (g·(φ-φ₀))²)"""
                    return f0 + np.sqrt((delta_f / 2) ** 2 + (g * (phi - phi0)) ** 2)

                # Initial parameter guesses
                f0_guess = np.mean(peak_freq_smooth)  # Center frequency
                delta_f_guess = np.std(peak_freq_smooth) * 0.1  # Small gap estimate
                g_guess = np.std(dfreq_dflux) * np.std(flux_bias_array)  # Coupling strength estimate
                phi0_guess = np.mean(flux_bias_array)  # Flux offset

                # Fit the hyperbolic function
                popt, _ = optimize.curve_fit(
                    hyperbolic_func,
                    flux_bias_array,
                    peak_freq_smooth,
                    p0=[f0_guess, delta_f_guess, g_guess, phi0_guess],
                    maxfev=5000,
                )

                hyperbolic_fit_params = {
                    "f0": float(popt[0]),  # Center frequency (Hz)
                    "delta_f": float(popt[1]),  # Gap size (Hz)
                    "g": float(popt[2]),  # Coupling strength (Hz/V)
                    "phi0": float(popt[3]),  # Flux offset (V)
                }
            except Exception as e:
                logging.warning(f"Failed to fit hyperbolic function for qubit {q_name}: {e}")
                hyperbolic_fit_params = None

            # Assess success: found at least one crossing
            success = len(crossing_flux_biases) > 0

            fit_results[q_name] = FitParameters(
                success=success,
                avoided_crossing_flux_biases=crossing_flux_biases,
                num_crossings=len(crossing_flux_biases),
                hyperbolic_fit_params=hyperbolic_fit_params,
            )

        except Exception as e:
            logging.warning(f"Error fitting qubit {q_name}: {e}")
            fit_results[q_name] = FitParameters(
                success=False,
                avoided_crossing_flux_biases=[],
                num_crossings=0,
                hyperbolic_fit_params=None,
            )

    # Add fit results to dataset for easy access in plotting
    qubit_names = [q.name for q in node.namespace["qubits"]]
    success_array = xr.DataArray(
        [fit_results[q].success for q in qubit_names],
        coords={"qubit": qubit_names},
    )
    num_crossings_array = xr.DataArray(
        [fit_results[q].num_crossings for q in qubit_names],
        coords={"qubit": qubit_names},
    )

    fit = fit.assign_coords(
        success=("qubit", success_array.data),
        num_crossings=("qubit", num_crossings_array.data),
    )

    # Store crossing positions as a JSON string for netCDF serialization
    # Convert dict to JSON string since netCDF can't serialize nested dicts
    # Ensure all values are native Python types (not numpy) for JSON serialization
    crossing_dict = {q: [float(x) for x in fit_results[q].avoided_crossing_flux_biases] for q in qubit_names}
    fit.attrs["avoided_crossing_flux_biases"] = json.dumps(crossing_dict)

    # Ensure smoothed_peak_frequency has proper coordinates
    if "smoothed_peak_frequency" in fit.data_vars:
        fit["smoothed_peak_frequency"] = fit["smoothed_peak_frequency"].assign_coords(
            qubit=qubit_names, flux_bias=fit.flux_bias
        )

    return fit, fit_results
