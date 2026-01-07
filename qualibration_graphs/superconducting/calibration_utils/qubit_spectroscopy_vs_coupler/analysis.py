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
    hyperbolic_fit_params: list[dict] | None = None
    """
    List of parameters for fitted avoided crossing models, one per crossing.
    Each dict contains: 'w_r' (resonator frequency), 'w_q0' (qubit frequency at phi0),
    'alpha' (qubit frequency slope), 'phi0' (crossing position), 'g' (coupling strength),
    'flux_range' (fit window), 'branch_assignments' (branch assignments for each data point)
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

            # Log hyperbolic fit information
            hyperbolic_fits = fit_results[q].get("hyperbolic_fit_params")
            if hyperbolic_fits is not None:
                num_fits = len(hyperbolic_fits)
                s_crossings += f" | {num_fits}/{num_crossings} hyperbolic fit(s) successful"
            else:
                s_crossings += " | No hyperbolic fits"
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

            # Fit hyperbolic function locally around each crossing
            # Hyperbolic form: f(φ) = f₀ + √((Δf/2)² + (g·(φ-φ₀))²)
            hyperbolic_fit_params = []

            if len(crossing_flux_biases) > 0:
                # Automatically calculate window size based on data characteristics
                # Calculate typical flux spacing
                if len(flux_bias_array) > 1:
                    flux_step = np.mean(np.diff(flux_bias_array))
                    flux_range_total = np.max(flux_bias_array) - np.min(flux_bias_array)
                else:
                    flux_step = 0.001  # Fallback
                    flux_range_total = 0.1  # Fallback

                # Target window: use ~30-40 data points, but adapt to data density
                target_points = 35
                base_window_volts = target_points * abs(flux_step)

                # Ensure minimum and maximum window sizes relative to scanning range
                # At least 5% of range or 15 points
                min_window_volts = max(0.05 * flux_range_total, 15 * abs(flux_step))
                max_window_volts = 0.5 * flux_range_total  # At most 50% of range
                base_window_volts = np.clip(base_window_volts, min_window_volts, max_window_volts)

                # Define avoided crossing model (2x2 eigenvalues, with linear bare tunable mode)
                def avoided_crossing(phi, w_r, w_q0, alpha, phi0, g, branch):
                    """
                    Avoided crossing model with both branches.
                    branch = +1 for upper eigenvalue, -1 for lower eigenvalue
                    """
                    w1 = w_r
                    w2 = w_q0 + alpha * (phi - phi0)
                    avg = 0.5 * (w1 + w2)
                    det = 0.5 * (w1 - w2)
                    split = np.sqrt(det**2 + g**2)
                    return avg + branch * split

                def model(xdata, w_r, w_q0, alpha, phi0, g):
                    """Model function for curve_fit"""
                    phi, br = xdata
                    return avoided_crossing(phi, w_r, w_q0, alpha, phi0, g, br)

                # Fit around each crossing
                for i, crossing_flux in enumerate(crossing_flux_biases):
                    try:
                        # Calculate adaptive window size for this crossing
                        # Consider distance to neighboring crossings
                        window_volts = base_window_volts

                        # If there are neighboring crossings, reduce window to avoid overlap
                        if len(crossing_flux_biases) > 1:
                            # Find distances to nearest neighbors
                            distances = []
                            for other_crossing in crossing_flux_biases:
                                if other_crossing != crossing_flux:
                                    distances.append(abs(other_crossing - crossing_flux))

                            if distances:
                                min_distance = min(distances)
                                # Use at most 60% of distance to nearest crossing, but not less than minimum
                                adaptive_window = min(0.6 * min_distance, base_window_volts)
                                window_volts = max(adaptive_window, min_window_volts)

                        # Find indices within the window around this crossing
                        window_mask = np.abs(flux_bias_array - crossing_flux) <= window_volts

                        # Extract data within window
                        flux_window = flux_bias_array[window_mask]
                        freq_window = peak_freq_smooth[window_mask]

                        # Ensure we have enough points for fitting
                        if len(flux_window) < 10:
                            logging.warning(
                                f"Insufficient data points ({len(flux_window)}) for avoided crossing fit "
                                f"around crossing at {crossing_flux * 1e3:.1f} mV for qubit {q_name}. Skipping."
                            )
                            continue

                        # Initial guesses (heuristics)
                        left = flux_window < crossing_flux
                        right = flux_window > crossing_flux

                        w_r0 = np.median(freq_window[left]) if np.any(left) else np.median(freq_window)
                        w_q00 = np.median(freq_window[right]) if np.any(right) else np.median(freq_window)

                        # Slope guess from right side (rough linear fit)
                        if np.sum(right) >= 5:
                            alpha0, _ = np.polyfit(flux_window[right], freq_window[right], 1)
                        elif np.sum(left) >= 5:
                            alpha0, _ = np.polyfit(flux_window[left], freq_window[left], 1)
                        else:
                            alpha0, _ = np.polyfit(flux_window, freq_window, 1)

                        # Crossing guess at the identified crossing position
                        phi00 = float(crossing_flux)

                        # Coupling guess from overall scale (kept >= 1 MHz)
                        g0 = max(1e6, 0.25 * (np.percentile(freq_window, 95) - np.percentile(freq_window, 5)))

                        p = np.array([w_r0, w_q00, alpha0, phi00, g0], dtype=float)

                        def fit_with_branches(flux_, freq_, branch_, p0):
                            """Fit model with given branch assignments"""
                            popt, _ = optimize.curve_fit(model, (flux_, branch_), freq_, p0=p0, maxfev=50000)
                            return popt

                        # EM-style loop: assign branch using current params, then refit
                        branch = np.ones_like(freq_window, dtype=float)
                        for iteration in range(30):
                            upper = avoided_crossing(flux_window, *p, branch=+1.0)
                            lower = avoided_crossing(flux_window, *p, branch=-1.0)

                            new_branch = np.where(
                                np.abs(freq_window - upper) <= np.abs(freq_window - lower), +1.0, -1.0
                            )

                            p_new = fit_with_branches(flux_window, freq_window, new_branch, p)

                            # Stop if stable
                            if np.all(new_branch == branch) and np.allclose(p_new, p, rtol=1e-5, atol=1e-3):
                                p = p_new
                                branch = new_branch
                                break

                            p = p_new
                            branch = new_branch

                        # Optional robust cleanup: drop large-residual outliers once, refit
                        upper = avoided_crossing(flux_window, *p, branch=+1.0)
                        lower = avoided_crossing(flux_window, *p, branch=-1.0)
                        pred = np.where(branch > 0, upper, lower)
                        resid = freq_window - pred

                        mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
                        sigma = 1.4826 * mad
                        keep = np.abs(resid) < 4.5 * sigma  # ~4.5-sigma rule

                        if np.sum(keep) >= 10:  # Ensure enough points after outlier removal
                            p_final = fit_with_branches(flux_window[keep], freq_window[keep], branch[keep], p)
                            # Recompute branch assignments with final parameters
                            upper_final = avoided_crossing(flux_window[keep], *p_final, branch=+1.0)
                            lower_final = avoided_crossing(flux_window[keep], *p_final, branch=-1.0)
                            branch_final = np.where(
                                np.abs(freq_window[keep] - upper_final) <= np.abs(freq_window[keep] - lower_final),
                                +1.0,
                                -1.0,
                            )
                        else:
                            p_final = p
                            branch_final = branch[keep] if np.any(keep) else branch

                        w_r, w_q0, alpha, phi0_fit, g = p_final

                        # Store fit parameters
                        fit_params = {
                            "w_r": float(w_r),  # Resonator frequency (Hz)
                            "w_q0": float(w_q0),  # Qubit frequency at phi0 (Hz)
                            "alpha": float(alpha),  # Qubit frequency slope (Hz/V)
                            "phi0": float(phi0_fit),  # Crossing flux position (V)
                            "g": float(g),  # Coupling strength (Hz)
                            "flux_range": [
                                float(np.min(flux_window)),
                                float(np.max(flux_window)),
                            ],  # Fit window range
                            "branch_assignments": (
                                branch_final.tolist() if hasattr(branch_final, "tolist") else branch_final
                            ),
                        }
                        hyperbolic_fit_params.append(fit_params)

                    except Exception as e:
                        logging.warning(
                            f"Failed to fit hyperbolic function around crossing at "
                            f"{crossing_flux * 1e3:.1f} mV for qubit {q_name}: {e}"
                        )
                        # Continue with other crossings
                        continue

            # Set to None if no fits were successful
            if len(hyperbolic_fit_params) == 0:
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

    # Store hyperbolic fit parameters as JSON string
    hyperbolic_fits_dict = {}
    for q in qubit_names:
        if fit_results[q].hyperbolic_fit_params is not None:
            # Convert each fit dict to ensure all values are native Python types
            hyperbolic_fits_dict[q] = []
            for fp in fit_results[q].hyperbolic_fit_params:
                fit_dict = {
                    "w_r": float(fp["w_r"]),
                    "w_q0": float(fp["w_q0"]),
                    "alpha": float(fp["alpha"]),
                    "phi0": float(fp["phi0"]),
                    "g": float(fp["g"]),
                    "flux_range": [float(fp["flux_range"][0]), float(fp["flux_range"][1])],
                }
                # Store branch assignments if available (may be large, so optional)
                if "branch_assignments" in fp:
                    fit_dict["branch_assignments"] = fp["branch_assignments"]
                hyperbolic_fits_dict[q].append(fit_dict)
        else:
            hyperbolic_fits_dict[q] = []
    fit.attrs["hyperbolic_fit_params"] = json.dumps(hyperbolic_fits_dict)

    # Ensure smoothed_peak_frequency has proper coordinates
    if "smoothed_peak_frequency" in fit.data_vars:
        fit["smoothed_peak_frequency"] = fit["smoothed_peak_frequency"].assign_coords(
            qubit=qubit_names, flux_bias=fit.flux_bias
        )

    return fit, fit_results
