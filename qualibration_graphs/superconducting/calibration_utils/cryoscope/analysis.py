from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation #, unwrap_phase
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit, minimize
from scipy.signal import deconvolve, savgol_filter, lfilter, convolve
from qualang_tools.digital_filters import calc_filter_taps


def transform_to_circle(x, y):
    def ellipse_residuals(params, x, y):
        a, b, cx, cy, angle = params
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = cos_angle * (x - cx) + sin_angle * (y - cy)
        y_rot = -sin_angle * (x - cx) + cos_angle * (y - cy)
        residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
        return np.sum(residuals**2)  # Return the sum of squared residuals

    # Fit ellipse to points
    initial_guess = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.0,
    ]  # Initial guess for ellipse parameters including angle
    result = minimize(ellipse_residuals, initial_guess, args=(x, y))
    a_fit, b_fit, cx_fit, cy_fit, angle_fit = result.x

    # Transform ellipse to circle
    scale_factor = max(a_fit, b_fit)
    a_circle = scale_factor
    b_circle = scale_factor
    cx_circle = cx_fit
    cy_circle = cy_fit

    # Rotate the ellipse points
    cos_angle = np.cos(angle_fit)
    sin_angle = np.sin(angle_fit)

    # Apply transform to xy points
    # Step 1: Rotate the original points to align with the ellipse's axes
    x_rot = cos_angle * (x - cx_fit) + sin_angle * (y - cy_fit)
    y_rot = -sin_angle * (x - cx_fit) + cos_angle * (y - cy_fit)

    # Step 2: Scale the rotated points to transform the ellipse into a circle
    x_scaled = x_rot / a_fit * a_circle
    y_scaled = y_rot / b_fit * b_circle

    # Step 3: Rotate the scaled points back to the original orientation
    x_transformed = cos_angle * x_scaled - sin_angle * y_scaled + cx_circle
    y_transformed = sin_angle * x_scaled + cos_angle * y_scaled + cy_circle

    return x_transformed, y_transformed


def savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x, range, order, deriv=0, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def diff_savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x / (2 * np.pi), range, order, deriv=1, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def cryoscope_frequency(ds, stable_time_indices, quad_term=-1, sg_range=3, sg_order=2):
    ds = ds.copy()

    freq_cryoscope = diff_savgol(ds, "time", range=sg_range, order=sg_order)

    ds["freq"] = freq_cryoscope

    flux_cryoscope = np.sqrt(np.abs(1e9 * freq_cryoscope / quad_term)).fillna(0)

    if quad_term == -1:
        flux_cryoscope = flux_cryoscope / flux_cryoscope.sel(
            time=slice(stable_time_indices[0], stable_time_indices[1])
        ).mean(dim="time")

    ds["flux"] = flux_cryoscope
    return ds


def expdecay(x, s, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t0: time shift
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))


def two_expdecay(x, s, a, t, a2, t2):
    """Double exponential decay defined as s * (1 + a * np.exp(-x / t) + a2 * np.exp(-x / t2)).
    :param x: numpy array for the time vector in ns
    :param s: float for the scaling factor
    :param a: float for the first exponential amplitude
    :param t: float for the first exponential decay time in ns
    :param a2: float for the second exponential amplitude
    :param t2: float for the second exponential decay time in ns
    :return: numpy array for the double exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t) + a2 * np.exp(-(x) / t2))


def single_exp(da, plot=True):
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(20, None)).mean().values
    print(first_vals, final_vals)

    fit = da.curvefit(
        "time",
        expdecay,
        p0={"a": 1 - first_vals / final_vals, "t": 50, "s": final_vals},
    ).curvefit_coefficients

    fit_vals = {k: v for k, v in zip(fit.to_dict()["coords"]["param"]["data"], fit.to_dict()["data"])}

    t_s = 1
    alpha = np.exp(-t_s / fit_vals["t"])
    A = fit_vals["a"]
    fir = [1 / (1 + A), -alpha / (1 + A)]
    iir = [(A + alpha) / (1 + A)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(da.time, da, label="data")
        ax.plot(da.time, expdecay(da.time, **fit_vals), label="fit")
        ax.grid("all")
        ax.legend()
        print(f"Qubit - FIR: {fir}\nIIR: {iir}")
    else:
        fig = None
        ax = None
    return fir, iir, fig, ax, (da.time, expdecay(da.time, **fit_vals))

def estimate_fir_coefficients(convolved_signal, step_response, num_coefficients):
    """
    Estimate the FIR filter coefficients from a convolved signal.

    :param convolved_signal: The signal after being convolved with the FIR filter.
    :param step_response: The original step response signal.
    :param num_coefficients: Number of coefficients of the FIR filter to estimate.
    :return: Estimated FIR coefficients.
    """
    # Deconvolve to estimate the impulse response
    estimated_impulse_response, _ = deconvolve(convolved_signal, step_response)

    # Truncate or zero-pad the estimated impulse response to match the desired number of coefficients
    if len(estimated_impulse_response) > num_coefficients:
        # Truncate if the estimated response is longer than the desired number of coefficients
        estimated_coefficients = estimated_impulse_response[:num_coefficients]
    else:
        # Zero-pad if shorter
        estimated_coefficients = np.pad(
            estimated_impulse_response,
            (0, num_coefficients - len(estimated_impulse_response)),
            "constant",
        )

    return estimated_coefficients


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def unwrap_phase(da, dim="time"):
    """
    Unwrap phase data along a specified dimension.

    Parameters:
    da : xr.DataArray - DataArray containing wrapped phase data.
    dim : str - Dimension along which to unwrap the phase.

    Returns:
    xr.DataArray - DataArray with unwrapped phase.
    """
    def unwrap_func(phase_array):
        return np.unwrap(phase_array)

    return xr.apply_ufunc(
        unwrap_func,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype]
    )

def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """
    Fit raw cryoscope data with exponential models.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing I/Q or state data
    node : QualibrationNode
        Node containing parameters and configuration

    Returns
    -------
    tuple
        (fitted_dataset, fit_results_dict)
    """
    try:
        if hasattr(ds, "I"):
            data = "I"
        elif hasattr(ds, "state"):
            data = "state"
        else:
            raise ValueError("Dataset must contain either 'I' or 'state' data")

        dafit = fit_oscillation(ds[data], "frame")

        daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")
        sg_order = 2
        sg_range = 3

        ds_fit = cryoscope_frequency(
            daphi,
            quad_term=-1,
            stable_time_indices=(node.parameters.cryoscope_len - 20, node.parameters.cryoscope_len),
            sg_order=sg_order,
            sg_range=sg_range,
        )

        first_vals = ds_fit.flux.sel(time=slice(0, 1)).mean()
        final_vals = ds_fit.flux.sel(time=slice(node.parameters.cryoscope_len - 20, None)).mean()

        qubit = node.namespace["qubits"][0].name

        # Find the index where ds_fit.flux is closest to 1/e
        qubit_flux = ds_fit.flux.sel(qubit=qubit)
        flux_vals = qubit_flux.values
        range_vals = final_vals - first_vals
        guess_val = 0.37 * range_vals + first_vals

        # Handle both scalar and array cases for guess_val
        if hasattr(guess_val, "data"):
            guess_val_scalar = float(guess_val.data)
        else:
            guess_val_scalar = float(guess_val)

        idx_closest = np.abs(flux_vals - guess_val_scalar).argmin()

        if node.parameters.exp_1_tau_guess is not None:
            initial_guess_tau = node.parameters.exp_1_tau_guess
        else:
            # Ensure the index is within bounds
            if idx_closest >= len(ds_fit.time.values):
                idx_closest = len(ds_fit.time.values) - 1
            initial_guess_tau = ds_fit.time.values[idx_closest]

        print(f"Initial guess for tau: {initial_guess_tau}")

        # Single exponential fit
        try:
            p0 = [final_vals, -1 + first_vals / final_vals, initial_guess_tau]
            fit, _ = curve_fit(expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
            fit1_success = True
        except Exception as e:
            fit = p0
            fit1_success = False
            print("single exp fit failed with error:\n", e)

        # Double exponential fit
        try:
            p0 = [fit[0], fit[1], initial_guess_tau, fit[1], fit[2]]
            fit2, _ = curve_fit(two_expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
            fit2_success = True
        except Exception as e:
            fit2 = []
            fit2_success = False
            print("two exp fit failed with error:\n", e)

        # Save fit results as attributes
        ds_fit.attrs["fit_1exp"] = fit
        ds_fit.attrs["fit_1exp_success"] = fit1_success
        ds_fit.attrs["fit_2exp"] = fit2
        ds_fit.attrs["fit_2exp_success"] = fit2_success

        ds["fit_results"] = ds_fit

        fit, fit_results = _extract_relevant_fit_parameters(ds, node)

        return fit, fit_results

    except Exception as e:
        print(f"Critical error in fit_raw_data: {e}")
        print("Returning minimal dataset with failed fit results")

        # Create a minimal failed result
        qubit_names = [q.name for q in node.namespace["qubits"]]
        fit_results = {qubit_name: FitParameters(success=False) for qubit_name in qubit_names}

        # Create a copy of the original dataset for ds_fit
        # Don't assign it to ds["fit_results"] as that causes the xarray error
        ds_fit = ds.copy()
        ds_fit.attrs["fit_1exp_success"] = False
        ds_fit.attrs["fit_2exp_success"] = False

        return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds: xr.Dataset, node: QualibrationNode):
    """Extract relevant fit parameters from the dataset and add metadata."""
    # Assess whether the fit was successful or not

    # Check if ds has fit_results (normal case) or use ds directly (error case)
    if "fit_results" in ds:
        fit = ds["fit_results"]
    else:
        fit = ds

    fit_results = {}

    # Get qubit names from the node if qubit dimension doesn't exist
    if hasattr(fit, "qubit") and hasattr(fit.qubit, "values"):
        qubit_names = fit.qubit.values
    else:
        qubit_names = [q.name for q in node.namespace["qubits"]]

    for q in qubit_names:
        fit1_success = fit.attrs.get("fit_1exp_success", False)
        fit2_success = fit.attrs.get("fit_2exp_success", False)

        # Determine overall success based on the number of exponents requested
        if node.parameters.number_of_exponents == 1:
            success = fit1_success
        elif node.parameters.number_of_exponents == 2:
            success = fit2_success
        else:
            success = fit1_success or fit2_success  # Default fallback

        fit_results[q] = FitParameters(
            success=success,
            fit1_success=fit1_success,
            fit2_success=fit2_success,
            fit1_A=fit.attrs.get("fit_1exp", [None, None, None])[1] if fit1_success else None,
            fit1_tau=fit.attrs.get("fit_1exp", [None, None, None])[2] if fit1_success else None,
            fit2_A1=(fit.attrs.get("fit_2exp", [None, None, None, None, None])[1] if fit2_success else None),
            fit2_tau1=(fit.attrs.get("fit_2exp", [None, None, None, None, None])[2] if fit2_success else None),
            fit2_A2=(fit.attrs.get("fit_2exp", [None, None, None, None, None])[3] if fit2_success else None),
            fit2_tau2=(fit.attrs.get("fit_2exp", [None, None, None, None, None])[4] if fit2_success else None),
        )
    return ds, fit_results


def log_fitted_results(fit_results: dict, log_callable=print):
    """Log the fitted results for each qubit.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing fit results for each qubit.
    log_callable : callable, optional
        Function to use for logging (default is print).
    """
    for qubit_name, fit_result in fit_results.items():
        log_callable(f"=== {qubit_name} ===")
        if fit_result.success:
            log_callable("Overall fit: SUCCESSFUL")
            if fit_result.fit1_success:
                log_callable("Single exponential fit: SUCCESS")
                if fit_result.fit1_A is not None and fit_result.fit1_tau is not None:
                    log_callable(f"  A = {fit_result.fit1_A:.4f}")
                    log_callable(f"  tau = {fit_result.fit1_tau:.2f} ns")
            else:
                log_callable("Single exponential fit: FAILED")

            if fit_result.fit2_success:
                log_callable("Double exponential fit: SUCCESS")
                if all(
                    val is not None
                    for val in [fit_result.fit2_A1, fit_result.fit2_tau1, fit_result.fit2_A2, fit_result.fit2_tau2]
                ):
                    log_callable(f"  A1 = {fit_result.fit2_A1:.4f}, tau1 = {fit_result.fit2_tau1:.2f} ns")
                    log_callable(f"  A2 = {fit_result.fit2_A2:.4f}, tau2 = {fit_result.fit2_tau2:.2f} ns")
            else:
                log_callable("Double exponential fit: FAILED")
        else:
            log_callable("Overall fit: FAILED")
            log_callable(f"Single exponential fit: {'SUCCESS' if fit_result.fit1_success else 'FAILED'}")
            log_callable(f"Double exponential fit: {'SUCCESS' if fit_result.fit2_success else 'FAILED'}")
        log_callable("")


@dataclass
class FitParameters:
    success: bool = False
    fit1_success: bool = False
    fit2_success: bool = False
    fit1_A: Optional[float] = None
    fit1_tau: Optional[float] = None
    fit2_A1: Optional[float] = None
    fit2_tau1: Optional[float] = None
    fit2_A2: Optional[float] = None
    fit2_tau2: Optional[float] = None



def estimate_fir_coefficients(convolved_signal, step_response, num_coefficients):
    """
    Estimate the FIR filter coefficients from a convolved signal.

    :param convolved_signal: The signal after being convolved with the FIR filter.
    :param step_response: The original step response signal.
    :param num_coefficients: Number of coefficients of the FIR filter to estimate.
    :return: Estimated FIR coefficients.
    """
    # Deconvolve to estimate the impulse response
    estimated_impulse_response, _ = deconvolve(convolved_signal, step_response)

    # Truncate or zero-pad the estimated impulse response to match the desired number of coefficients
    if len(estimated_impulse_response) > num_coefficients:
        # Truncate if the estimated response is longer than the desired number of coefficients
        estimated_coefficients = estimated_impulse_response[:num_coefficients]
    else:
        # Zero-pad if shorter
        estimated_coefficients = np.pad(
            estimated_impulse_response, (0, num_coefficients - len(estimated_impulse_response)), "constant"
        )

    return estimated_coefficients

def expdecay(x, s, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t0: time shift
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))

@dataclass
class AdvancedFitParameters:
    """Stores the advanced cryoscope analysis parameters for a single qubit"""
    success: bool = False
    rise_index: Optional[int] = None
    drop_index: Optional[int] = None
    fit_parameters: Optional[Dict] = None
    fir_coefficients: Optional[List[float]] = None
    exponential_filter: Optional[List] = None
    convolved_fir: Optional[List[float]] = None
    iir_coefficients: Optional[List[float]] = None
    final_vals: Optional[float] = None


def calculate_advanced_filters(ds_fit: xr.Dataset, node: QualibrationNode) -> Dict[str, AdvancedFitParameters]:
    """
    Calculate advanced FIR and IIR filters for cryoscope analysis.
    
    Parameters
    ----------
    ds_fit : xr.Dataset
        Dataset containing the fitted cryoscope data
    node : QualibrationNode
        Node containing parameters and configuration
        
    Returns
    -------
    Dict[str, AdvancedFitParameters]
        Dictionary containing advanced fit parameters for each qubit
    """
    advanced_results = {}
    
    for qubit in node.namespace["qubits"]:
        qubit_name = qubit.name
        print(f"\n=== Processing {qubit_name} ===")
        
        # Get the fit results for this qubit
        if qubit_name not in node.results.get("fit_results", {}):
            print(f"No fit results available for {qubit_name}")
            continue
            
        fit_result = node.results["fit_results"][qubit_name]
        
        # Check if we have successful fits
        if not fit_result.get("success", False):
            print(f"Fit failed for {qubit_name}, skipping advanced analysis")
            continue
            
        # Get the flux data from the fit results
        if hasattr(ds_fit, 'fit_results') and hasattr(ds_fit.fit_results, 'flux'):
            flux_data = ds_fit.fit_results.flux.sel(qubit=qubit_name)
        else:
            print(f"No flux data available for {qubit_name}")
            continue
        
        try:
            # Set rise and drop indices for analysis
            print('\033[1m\033[32m SETTING RISE AND DROP INDICES \033[0m')
            threshold = flux_data.max().values * 0.6  # Set the threshold value
            rise_index = np.argmax(flux_data.values > threshold) + 1
            drop_index = len(flux_data) - 0
            
            # Extract the rising part of the data for analysis
            flux_cryoscope_tp = flux_data.sel(time=slice(rise_index, drop_index))
            flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
                time=flux_cryoscope_tp.time - rise_index + 1)
            
            # Get initial and final values for exponential fit
            first_vals = flux_cryoscope_tp.sel(time=slice(0, 1)).mean().values
            final_vals = flux_cryoscope_tp.sel(time=slice(-20, None)).mean().values
            
            # Exponential fit
            print('\033[1m\033[32m EXPONENTIAL FIT \033[0m')
            exponential_fit_time_interval = [1, 30]
            time_slice = flux_cryoscope_tp.time.sel(time=slice(*exponential_fit_time_interval))
            start_index, end_index = time_slice.time.values[0], time_slice.time.values[-1]
            
            p0 = [final_vals, -1 + first_vals / final_vals, 50]
            fit, pcov, infodict, errmsg, ier = curve_fit(
                expdecay, 
                flux_cryoscope_tp.time[start_index:end_index], 
                flux_cryoscope_tp[start_index:end_index],
                p0=p0, 
                maxfev=10000, 
                ftol=1e-8, 
                full_output=True
            )
            
            # Calculate residuals and print fit information
            y_fit = expdecay(flux_cryoscope_tp.time, *fit)
            residuals = flux_cryoscope_tp - y_fit
            chi_squared = np.sum(residuals**2)
            print(f"\nSingle Exponential Fit Results for {qubit_name}:")
            print(f"Number of iterations: {infodict['nfev']}")
            print(f"Final chi-squared: {chi_squared:.6f}")
            print(f"RMS of residuals: {np.sqrt(np.mean(residuals**2)):.6f}")
            print(f"Fit parameters: {fit}")
            print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")
            
            # Calculate filter response
            print('\033[1m\033[32m CALCULATE FILTERED RESPONSE \033[0m')
            exponential_filter = list(zip([fit[1] * 1.0], [fit[2]]))
            feedforward_taps_1exp, feedback_tap_1exp = calc_filter_taps(exponential=exponential_filter)
            
            FIR_1exp = feedforward_taps_1exp
            IIR_1exp = [1, -feedback_tap_1exp[0]]
            
            # Apply filter to the flux data
            flux_cryoscope_filtered = flux_data.copy()
            flux_cryoscope_filtered.values[0] = 0  # Set first point to zero
            filtered_response_long_1exp = lfilter(FIR_1exp, IIR_1exp, flux_cryoscope_filtered.values)
            
            # Calculate FIR filter for short time-scale corrections
            print('\033[1m\033[32m CALCULATE FIR FILTER \033[0m')
            response_long = filtered_response_long_1exp[1:]
            flux_q = flux_data[1:].copy()
            flux_q.values = response_long
            flux_q_tp = flux_q.sel(time=slice(rise_index, 200))  # Use first 200 ns for FIR calculation
            flux_q_tp = flux_q_tp.assign_coords(time=flux_q_tp.time - rise_index)
            final_vals = flux_q_tp.sel(time=slice(-20, None)).mean().values
            
            # Create step function for FIR estimation
            step = np.ones(len(flux_q) + 100) * final_vals
            fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 24)
            
            # Combine FIR and IIR filters
            convolved_fir = convolve(FIR_1exp, fir_est, mode='full')
            
            # Store the advanced results
            advanced_results[qubit_name] = AdvancedFitParameters(
                success=True,
                rise_index=rise_index,
                drop_index=drop_index,
                fit_parameters={
                    's': fit[0],
                    'a': fit[1], 
                    't': fit[2]  # Changed from 'tau' to 't' to match expdecay function signature
                },
                fir_coefficients=fir_est.tolist(),
                exponential_filter=exponential_filter,
                convolved_fir=convolved_fir.tolist(),
                iir_coefficients=IIR_1exp,
                final_vals=final_vals
            )
            
            print(f"\nFilter coefficients for {qubit_name}:")
            print(f"FIR coefficients: {fir_est.tolist()}")
            print(f"Exponential filter: {exponential_filter}")
            print(f"IIR coefficients: {IIR_1exp}")
            
        except Exception as e:
            print(f"Advanced analysis failed for {qubit_name}: {e}")
            advanced_results[qubit_name] = AdvancedFitParameters(success=False)
            continue
    
    print('\033[1m\033[32m ADVANCED ANALYSIS COMPLETE \033[0m')
    return advanced_results





def log_advanced_results(advanced_results: Dict[str, AdvancedFitParameters], log_callable=print):
    """Log the advanced analysis results for each qubit.
    
    Parameters
    ----------
    advanced_results : Dict[str, AdvancedFitParameters]
        Dictionary containing advanced fit parameters for each qubit
    log_callable : callable, optional
        Function to use for logging (default is print).
    """
    for qubit_name, result in advanced_results.items():
        log_callable(f"=== Advanced Analysis Results for {qubit_name} ===")
        if result.success:
            log_callable("Advanced analysis: SUCCESSFUL")
            log_callable(f"Rise index: {result.rise_index}")
            log_callable(f"Drop index: {result.drop_index}")
            if result.fit_parameters:
                log_callable(f"Fit parameters: s={result.fit_parameters['s']:.6f}, a={result.fit_parameters['a']:.6f}, t={result.fit_parameters['t']:.6f}")
            if result.fir_coefficients:
                log_callable(f"FIR coefficients: {result.fir_coefficients[:5]}... (showing first 5)")
            if result.exponential_filter:
                log_callable(f"Exponential filter: {result.exponential_filter}")
            if result.iir_coefficients:
                log_callable(f"IIR coefficients: {result.iir_coefficients}")
        else:
            log_callable("Advanced analysis: FAILED")
        log_callable("")
