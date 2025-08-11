
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence
from functools import reduce

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_decay_exp, fit_oscillation, peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit, minimize
from numpy.polynomial import Polynomial as P


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

#############################################################################################################
###                                  Cascading analysis functions                                         ###
#############################################################################################################


@dataclass
class CascadeFitParameters:
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

def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for fitting spectroscopy peaks.
    
    Args:
        x (array): X-axis values
        a (float): Amplitude
        x0 (float): Center position
        sigma (float): Width parameter
        offset (float): Vertical offset
    
    Returns:
        array: Gaussian values
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def fit_gaussian(freqs, states):
    """Fit Gaussian to spectroscopy data and return center frequency.
    
    Args:
        freqs (array): Frequency points
        states (array): Measured states
        
    Returns:
        float: Center frequency or np.nan if fit fails
    """
    p0 = [
        np.max(states) - np.min(states),   # amplitude
        freqs[np.argmax(states)],          # center
        (freqs[-1] - freqs[0]) / 10,        # width
        np.min(states)                     # offset
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return popt[1]  # center frequency
    except RuntimeError:
        return np.nan

def single_exp_decay(t, amp, tau):
    """Single exponential decay without offset
    
    Args:
        t (array): Time points
        amp (float): Amplitude of the decay
        tau (float): Time constant of the decay
        
    Returns:
        array: Exponential decay values
    """
    return amp * np.exp(-t/tau)

def sequential_exp_fit(t, y, start_fractions, verbose=True):
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term from the tail of the data
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        start_fractions (list): List of fractions (0 to 1) indicating where to start fitting each component
        verbose (bool): Whether to print detailed fitting information
        
    Returns:
        tuple: (components, a_dc, residual) where:
            - components: List of (amplitude, tau) pairs for each fitted component
            - a_dc: Fitted constant term
            - residual: Residual after subtracting all components
    """
    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0
    
    # Find the flat region in the tail by looking at local variance
    window = max(5, len(y) // 20)  # Window size by dividing signal into 20 equal pieces or at least 5 points
    rolling_var = np.array([np.var(y[i:i+window]) for i in range(len(y)-window)])
    # Find where variance drops below threshold, indicating flat region
    var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
    try:
        flat_start = np.where(rolling_var < var_threshold)[0][-1]
        # Use the flat region to estimate constant term
        a_dc = np.mean(y[flat_start:])
    except IndexError:
        print("No flat region found, using last point of the signal as constant term")
        a_dc = y[-1]

    if verbose:
        print(f"\nFitted constant term: {a_dc:.3e}")
    
    y_residual = y.copy() - a_dc
    
    for i, start_frac in enumerate(start_fractions):
        # Calculate start index for this component
        start_idx = int(len(t) * start_frac)
        if verbose:
            print(f"\nFitting component {i+1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")
        
        # Fit current component
        try:
            # Initial guess for parameters
            p0 = [
                y_residual[start_idx],  # amplitude
                t_offset[start_idx] / 3  # tau
            ]
            
            # Set bounds for the fit
            bounds = (
                [-np.inf, 0.1],  # lower bounds: amplitude can be negative, tau must be positive (0.1 ns is arbitrary)
                [np.inf, np.inf]  # upper bounds
            )
            
            # Perform the fit on the current interval
            t_fit = t_offset[start_idx:]
            y_fit = y_residual[start_idx:]
            popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)
            
            # Store the components
            amp, tau = popt
            components.append((amp, tau))
            if verbose:
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns")
            
            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset/tau)
            
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i+1}: {e}")
            break
    
    return components, a_dc, y_residual

def optimize_start_fractions(t, y, base_fractions, bounds_scale=0.5, verbose=True):
    """
    Optimize the start_fractions by minimizing the RMS between the data and the fitted sum 
    of exponentials using scipy.optimize.minimize.
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        base_fractions (list): Initial guess for start fractions
        bounds_scale (float): Scale factor for bounds around base fractions (0.5 means ±50%)
        
    Returns:
        tuple: (best_fractions, best_components, best_dc, best_rms)
    """
    
    def objective(x):
        """
        Objective function to minimize: RMS between the data and the fitted sum of 
        exponentials.
        """
        # Ensure fractions are ordered in descending order
        if not np.all(np.diff(x) < 0):
            return 1e6  # Return large value if constraint is violated
                
        components, _, residual = sequential_exp_fit(t, y, x, verbose=False)  # Always use verbose=False in objective
        if len(components) == len(base_fractions):
            current_rms = np.sqrt(np.mean(residual**2))
        else:
            current_rms = 1e6 # Return large value if fitting fails
            
        return current_rms
    
    # Define bounds for optimization
    bounds = []
    for base in base_fractions:
        min_val = base * (1 - bounds_scale)
        max_val = base * (1 + bounds_scale)
        bounds.append((min_val, max_val))
    
    if verbose:
        print("\nOptimizing start_fractions using scipy.optimize.minimize...")
        print(f"Initial values: {[f'{f:.5f}' for f in base_fractions]}")
        print(f"Bounds: ±{bounds_scale*100}% around initial values")
    
    # Run optimization
    result = minimize(
        objective,
        x0=base_fractions,
        bounds=bounds,
        method='Nelder-Mead',  # This method works well for non-smooth functions
        options={'disp': False, 'maxiter': 200}  # Set disp=False to reduce output
    )
    
    # Get final results
    if result.success:
        best_fractions = result.x
        components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, verbose=verbose)
        best_rms = np.sqrt(np.mean(best_residual**2))
        
        if verbose:
            print("\nOptimization successful!")
            print(f"Initial fractions: {[f'{f:.5f}' for f in base_fractions]}")
            print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
            print(f"Final RMS: {best_rms:.3e}")
            print(f"Number of iterations: {result.nit}")
    else:
        if verbose:
            print("\nOptimization failed. Using initial values.")
        best_fractions = base_fractions
        components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, verbose=verbose)
        best_rms = np.sqrt(np.mean(best_residual**2))
    
    return result.success, best_fractions, components, a_dc, best_rms


def fit_raw_data_cascade(ds: xr.Dataset, node: QualibrationNode):
    """
    1) Generate center frequencies for each qubit -> calculate flux response
    2) Fit exponential to flux response using optimize_start_fraction -> store in node.results["fit_results_cascade"]
        -Need to have success/fail, best fractions, best_components, best_a_dc, best_rms
    
    Note: This function works with both spectroscopy data (with 'state' variable) and I/Q data.
    For I/Q data, it uses the amplitude to estimate frequency shifts.
    """

    # Check if this is spectroscopy data (has 'state' variable) or I/Q data
    if 'state' in ds.data_vars:
        # This is spectroscopy data - proceed with cascade analysis
        print("Processing spectroscopy data with 'state' variable")
        freqs = ds['detuning'].values
        stacked = ds.transpose('qubit', 'time', 'detuning')

        center_freqs = xr.apply_ufunc(
            lambda states: fit_gaussian(freqs, states),
            stacked,
            input_core_dims=[['detuning']],
            output_core_dims=[[]],  # no dimensions left after fitting
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        ).rename({"state": "center_frequency"})

        flux_response = np.sqrt(center_freqs / xr.DataArray([q.freq_vs_flux_01_quad_term for q in node.namespace["qubits"]], coords={"qubit": center_freqs.qubit}, dims=["qubit"]))

        ds['center_freqs'] = center_freqs
        ds['flux_response'] = flux_response

        fit_results = {}
        for q in node.namespace["qubits"]:
            t_data = flux_response.sel(qubit=q.name).time.values
            y_data = flux_response.sel(qubit=q.name).values
            fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
                t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
            )
            fit_results[q.name] = {
                "fit_successful": fit_successful,
                "best_fractions": best_fractions,
                "best_components": best_components,
                "best_a_dc": best_a_dc,
                "best_rms": best_rms
            }
    else:
        # This is I/Q data from pi_flux experiment - analyze I/Q data to extract frequency information
        print("Processing I/Q data - extracting frequency information from quadratures")
        
        # For I/Q data, we need to extract frequency information from the quadratures
        # The peak in I/Q data corresponds to the resonant frequency
        if 'I' in ds.data_vars:
            # Use I quadrature to find frequency peaks
            iq_data = ds['I']
        elif 'IQ_abs' in ds.data_vars:
            # Use amplitude if available
            iq_data = ds['IQ_abs']
        else:
            print("Warning: No I/Q data found for cascade analysis")
            # Create dummy results
            center_freqs = xr.DataArray(
                np.zeros((len(ds.qubit), len(ds.time))),
                coords={"qubit": ds.qubit, "time": ds.time},
                dims=["qubit", "time"]
            )
            flux_response = xr.DataArray(
                np.zeros((len(ds.qubit), len(ds.time))),
                coords={"qubit": ds.qubit, "time": ds.time},
                dims=["qubit", "time"]
            )
            
            ds['center_freqs'] = center_freqs
            ds['flux_response'] = flux_response
            
            fit_results = {}
            for q in node.namespace["qubits"]:
                fit_results[q.name] = {
                    "fit_successful": False,
                    "best_fractions": [0.4, 0.15, 0.05],
                    "best_components": [(0.0, 100.0)],
                    "best_a_dc": 0.0,
                    "best_rms": np.nan
                }
            return ds, fit_results
        
        # Extract center frequencies from I/Q data by finding peaks along detuning dimension
        detuning_vals = ds['detuning'].values
        stacked = iq_data.transpose('qubit', 'time', 'detuning')
        
        # Fit Gaussian to find center frequency for each qubit and time point
        center_freqs = xr.apply_ufunc(
            lambda iq_slice: fit_gaussian(detuning_vals, iq_slice),
            stacked,
            input_core_dims=[['detuning']],
            output_core_dims=[[]],  # no dimensions left after fitting
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )
        
        # Calculate flux response from center frequencies
        # For I/Q data, we'll use the frequency shift as a proxy for flux response
        # Normalize by the maximum frequency shift
        freq_shift = center_freqs - center_freqs.min(dim='time')
        flux_response = freq_shift / freq_shift.max(dim='time') if freq_shift.max(dim='time') > 0 else freq_shift
        
        # Additional normalization to ensure reasonable scaling
        # Normalize each qubit's flux response to have a DC component close to 1
        for i, q in enumerate(node.namespace["qubits"]):
            qb_flux = flux_response.sel(qubit=q.name)
            # Use the last 20% of data as baseline
            baseline_idx = int(0.8 * len(qb_flux.time))
            baseline = qb_flux.isel(time=slice(baseline_idx, None)).mean()
            if baseline > 0:
                flux_response.loc[dict(qubit=q.name)] = qb_flux / baseline
        
        ds['center_freqs'] = center_freqs
        ds['flux_response'] = flux_response
        
        # Perform exponential fitting on the flux response data
        fit_results = {}
        for q in node.namespace["qubits"]:
            t_data = flux_response.sel(qubit=q.name).time.values
            y_data = flux_response.sel(qubit=q.name).values
            
            # Skip fitting if data is all zeros or NaN
            if np.all(y_data == 0) or np.any(np.isnan(y_data)):
                fit_results[q.name] = {
                    "fit_successful": False,
                    "best_fractions": [0.4, 0.15, 0.05],
                    "best_components": [(0.0, 100.0)],
                    "best_a_dc": 0.0,
                    "best_rms": np.nan
                }
            else:
                try:
                    fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
                        t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
                    )
                    fit_results[q.name] = {
                        "fit_successful": fit_successful,
                        "best_fractions": best_fractions,
                        "best_components": best_components,
                        "best_a_dc": best_a_dc,
                        "best_rms": best_rms
                    }
                except Exception as e:
                    print(f"Fit failed for qubit {q.name}: {e}")
                    fit_results[q.name] = {
                        "fit_successful": False,
                        "best_fractions": [0.4, 0.15, 0.05],
                        "best_components": [(0.0, 100.0)],
                        "best_a_dc": 0.0,
                        "best_rms": np.nan
                    }

    return ds, fit_results
    


def add_rational_terms(terms: List[Tuple[np.array, np.array]]) -> Tuple[np.array, np.array]:
    # Convert to Polynomial objects
    rational_terms = [(P(num), P(den)) for num, den in terms]

    # Compute common denominator
    common_den = reduce(lambda acc, t: acc * t[1], rational_terms, P([1]))

    # Adjust numerators to have the common denominator
    adjusted_numerators = []
    for num, den in rational_terms:
        multiplier = common_den // den
        adjusted_numerators.append(num * multiplier)

    # Sum all adjusted numerators
    final_numerator = sum(adjusted_numerators, P([0]))

    # Return as coefficient lists
    return final_numerator.coef, common_den.coef

def get_rational_filter_single_exp_cont_time(A: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1, 1/tau])
    b = np.array([A])
    return b, a

def decompose_exp_sum_to_cascade(A: Sequence, tau: Sequence, A_dc: float=1., Ts: float=0.5) -> \
        tuple[np.ndarray, np.ndarray, float]:
    """decompose_exp_sum_to_cascade
    Translate from filters configuration as defined in QUA for version 3.5 (sum of exponents) to the
    definition of version 3.4.1 (cascade of single exponents filters).
    In v3.5, the analog linear distortion H is characterized by step response:
    s_H(t) = (A_dc + sum(A[i] * exp(-t/tau[i]), for i in 0...(N-1)))*u(t)
    In v3.4.1, it is a cascade of single exponent filters, each with step response:
    s_H_i(t) = (1 + A_c[i] * exp(-t/tau_c[i]))*u(t)
    The parameters [(A_c[0], tau_c[0]), ...] are the definitions of the filters (under "exponents")
    in 3.4.1.
    To make the filters equivalent, the 3.4.1 cascade needs to scaled by the parameter scale.
    This scaling can be done by multiplying the FIR coefficients by scale, or by scaling the waverform
    amp accordingly.
    :return A_c, tau_c, scale
    """

    assert A_dc > 0.2, "HPF mode is currently not supported"

    ba_sum = [get_rational_filter_single_exp_cont_time(A_i, tau_i) for A_i, tau_i in zip(A, tau)]
    ba_sum += [([A_dc], [1])]

    b, a = add_rational_terms(ba_sum)

    # Check for numerical issues in polynomial coefficients
    if np.any(np.abs(b) > 1e10) or np.any(np.abs(a) > 1e10):
        print(f"  WARNING: Very large polynomial coefficients detected!")
        print(f"  This indicates numerical instability in the rational function addition.")
    
    if len(b) == 0 or len(a) == 0:
        print(f"  ERROR: Empty polynomial coefficients!")
        raise ValueError("Empty polynomial coefficients")

    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))

    # Add diagnostic information
    print(f"DEBUG: Polynomial coefficients:")
    print(f"  b (numerator): {b}")
    print(f"  a (denominator): {a}")
    print(f"  zeros: {zeros}")
    print(f"  poles: {poles}")
    print(f"  zeros are real: {np.all(np.isreal(zeros))}")
    print(f"  poles are real: {np.all(np.isreal(poles))}")
    
    if not np.all(np.isreal(zeros)):
        print(f"  Complex zeros found: {zeros[np.imag(zeros) != 0]}")
    if not np.all(np.isreal(poles)):
        print(f"  Complex poles found: {poles[np.imag(poles) != 0]}")

    assert np.all(np.isreal(zeros)) and np.all(np.isreal(poles)), \
        "Got complex zeros; this configuration can't be inverted or decomposed to cascade of single pole stages"

    tau_c = -1 / poles
    A_c = poles/zeros - 1

    scale = 1/A_dc


    return A_c, tau_c, scale
