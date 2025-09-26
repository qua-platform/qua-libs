from dataclasses import dataclass
from typing import Optional

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, unwrap_phase
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit, minimize
from scipy.signal import deconvolve, savgol_filter


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

        qubit = node.namespace["qubits"][0].name  # TODO: support multiple qubits

        # Find the index where ds_fit.flux is closest to 1/e
        qubit_flux = ds_fit.flux.sel(qubit=qubit)
        flux_vals = qubit_flux.values
        time_vals = ds_fit.time.values

        fitting_start_fractions = node.parameters.exponential_fit_time_fractions
        success, best_fractions, components, a_dc, best_rms = optimize_start_fractions(
            time_vals, flux_vals, fitting_start_fractions
        )
        # range_vals = final_vals - first_vals
        # guess_val = 0.37 * range_vals + first_vals

        # # Handle both scalar and array cases for guess_val
        # if hasattr(guess_val, "data"):
        #     guess_val_scalar = float(guess_val.data)
        # else:
        #     guess_val_scalar = float(guess_val)

        # idx_closest = np.abs(flux_vals - guess_val_scalar).argmin()

        # if node.parameters.exp_1_tau_guess is not None:
        #     initial_guess_tau = node.parameters.exp_1_tau_guess
        # else:
        #     # Ensure the index is within bounds
        #     if idx_closest >= len(ds_fit.time.values):
        #         idx_closest = len(ds_fit.time.values) - 1
        #     initial_guess_tau = ds_fit.time.values[idx_closest]

        # print(f"Initial guess for tau: {initial_guess_tau}")

        # # Single exponential fit
        # try:
        #     p0 = [final_vals, -1 + first_vals / final_vals, initial_guess_tau]
        #     fit, _ = curve_fit(expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
        #     fit1_success = True
        # except Exception as e:
        #     fit = p0
        #     fit1_success = False
        #     print("single exp fit failed with error:\n", e)

        # # Double exponential fit
        # try:
        #     p0 = [fit[0], fit[1], initial_guess_tau, fit[1], fit[2]]
        #     fit2, _ = curve_fit(two_expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
        #     fit2_success = True
        # except Exception as e:
        #     fit2 = None
        #     fit2_success = False
        #     print("two exp fit failed with error:\n", e)

        # # Save fit results as attributes
        # ds_fit.attrs["fit_1exp"] = fit
        # ds_fit.attrs["fit_1exp_success"] = fit1_success
        # ds_fit.attrs["fit_2exp"] = fit2
        # ds_fit.attrs["fit_2exp_success"] = fit2_success

        # ds["fit_results"] = ds_fit

        # fit, fit_results = _extract_relevant_fit_parameters(ds, node)

        # Save fit results as attributes (ensure NetCDF compatibility: only 1D arrays / scalars)
        ds_fit.attrs["fit_success"] = success
        if components is not None:
            try:
                amps = [float(a) for a, _ in components]
                taus = [float(t) for _, t in components]
            except Exception:
                amps, taus = [], []
            ds_fit.attrs["fit_component_amps"] = np.array(amps)
            ds_fit.attrs["fit_component_taus_ns"] = np.array(taus)
        ds_fit.attrs["fit_a_dc"] = float(a_dc) if a_dc is not None else np.nan

        ds["fit_results"] = ds_fit

        fit, fit_results = _extract_relevant_fit_parameters_new(ds, node)

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


def _extract_relevant_fit_parameters_new(ds: xr.Dataset, node: QualibrationNode):
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
        success = fit.attrs.get("fit_success", False)
        # Reconstruct components from stored 1D arrays if available
        if "fit_component_amps" in fit.attrs and "fit_component_taus_ns" in fit.attrs:
            amps = fit.attrs.get("fit_component_amps", [])
            taus = fit.attrs.get("fit_component_taus_ns", [])
            try:
                components = list(zip(list(amps), list(taus)))
            except Exception:
                components = []
        else:
            # Backward compatibility (older in-memory attribute, not NetCDF-safe but maybe present at runtime)
            components = fit.attrs.get("fit_components", [])
        a_dc = fit.attrs.get("fit_a_dc", None)

        fit_results[q] = FitParametersNEW(success=success, components=components, a_dc=a_dc)
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
        if getattr(fit_result, "success", False):
            log_callable("Overall fit: SUCCESSFUL")
        else:
            log_callable("Overall fit: FAILED")

        # New logging for FitParametersNEW structure
        if hasattr(fit_result, "components") and fit_result.components is not None:
            components = fit_result.components
            a_dc = getattr(fit_result, "a_dc", None)
            if a_dc is not None:
                log_callable(f"  DC term (a_dc): {a_dc:.6g}")
            if isinstance(components, (list, tuple)) and len(components) > 0:
                log_callable("  Exponential components (amplitude, tau [ns]):")
                for idx, comp in enumerate(components, start=1):
                    try:
                        amp, tau = comp
                        if a_dc not in (None, 0):
                            log_callable(f"    #{idx}: amp = {amp:.6g} (rel {amp / a_dc:.3f}), tau = {tau:.3f} ns")
                        else:
                            log_callable(f"    #{idx}: amp = {amp:.6g}, tau = {tau:.3f} ns")
                    except Exception:
                        log_callable(f"    #{idx}: {comp}")
            else:
                log_callable("  No exponential components fitted.")
        else:
            # Backwards compatibility: old FitParameters style
            if hasattr(fit_result, "fit1_success") or hasattr(fit_result, "fit2_success"):
                if getattr(fit_result, "fit1_success", False):
                    A = getattr(fit_result, "fit1_A", None)
                    tau = getattr(fit_result, "fit1_tau", None)
                    if A is not None and tau is not None:
                        log_callable(f"  Single exp: A = {A:.6g}, tau = {tau:.3f} ns")
                if getattr(fit_result, "fit2_success", False):
                    A1 = getattr(fit_result, "fit2_A1", None)
                    tau1 = getattr(fit_result, "fit2_tau1", None)
                    A2 = getattr(fit_result, "fit2_A2", None)
                    tau2 = getattr(fit_result, "fit2_tau2", None)
                    if None not in (A1, tau1, A2, tau2):
                        log_callable(
                            f"  Double exp: A1 = {A1:.6g}, tau1 = {tau1:.3f} ns | A2 = {A2:.6g}, tau2 = {tau2:.3f} ns"
                        )
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


@dataclass
class FitParametersNEW:
    # List of (amplitude, tau) tuples for each exponential component
    components: list
    # Constant (DC) term
    a_dc: float
    # Overall success flag
    success: bool = False


from typing import List, Tuple

# (Re)used below, already imported earlier in file; avoid duplicate imports
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit, minimize


def single_exp_decay(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    """Single exponential decay without offset

    Args:
        t (array): Time points
        amp (float): Amplitude of the decay
        tau (float): Time constant of the decay

    Returns:
        array: Exponential decay values
    """
    return amp * np.exp(-t / tau)


def sequential_exp_fit(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: List[float],
    fixed_taus: List[float] = None,
    a_dc: float = None,
    verbose: bool = 1,
) -> Tuple[List[Tuple[float, float]], float, np.ndarray]:
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term from the tail of the data
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components

    Args:
        t (array): Time points in nanoseconds, representing the time resolution of the pulse.
        y (array): Amplitude values of the pulse in volts.
    start_fractions (list): Fractions (0-1) where each component fit starts (user defined ordering).
        fixed_taus (list, optional): Fixed tau values (in nanoseconds) for each exponential component.
                                   If provided, only amplitudes are fitted, taus are constrained.
                                   Must have same length as start_fractions.
        a_dc (float, optional): Fixed constant term. If provided, the constant term is not fitted.
    verbose (int): Verbosity (0: silent, 1: summary, 2: detailed step-by-step info).

    Returns:
        tuple: (components, a_dc, residual) where:
            - components: List of (amplitude, tau) pairs for each fitted component
            - a_dc: Fitted constant term or the fixed constant term
            - residual: Data minus fitted curve after subtracting all components.
    """

    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0

    # Find the flat region in the tail by looking at local variance
    window = max(5, len(y) // 20)  # Window size by dividing signal into 20 equal pieces or at least 5 points
    rolling_var = np.array([np.var(y[i : i + window]) for i in range(len(y) - window)])
    # Find where variance drops below threshold, indicating flat region
    var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
    if a_dc is None:
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
            print(f"\nFitting component {i + 1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")

        # Fit current component
        try:
            # Prepare fitting parameters based on whether tau is fixed
            if fixed_taus is not None:
                # Use fixed tau - only fit amplitude using lambda
                tau_fixed = fixed_taus[i]
                p0 = [y_residual[start_idx]]  # Only amplitude initial guess
                if verbose:
                    print(f"Using fixed tau = {tau_fixed:.3f} ns")

                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(lambda t, amp: single_exp_decay(t, amp, tau_fixed), t_fit, y_fit, p0=p0)

                # Store the components
                amp = popt[0]
                tau = tau_fixed
            else:
                # Fit both amplitude and tau (original behavior)
                p0 = [y_residual[start_idx], t_offset[start_idx] / 3]  # amplitude  # tau

                # Set bounds for the fit
                bounds = (
                    [-np.inf, 0.1],
                    # lower bounds: amplitude can be negative, tau must be positive (0.1 ns is arbitrary)
                    [np.inf, np.inf],  # upper bounds
                )

                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)

                # Store the components
                amp, tau = popt

            components.append((amp, tau))
            if verbose:
                tau_status = "(fixed)" if fixed_taus is not None else ""
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns {tau_status}")

            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset / tau)

        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i + 1}: {e}")
            break

    return components, a_dc, y_residual


def optimize_start_fractions(t, y, start_fractions, bounds_scale=0.5, fixed_taus=None, a_dc=None, verbose=1):
    """
        Optimize the start fractions for a sum of exponentials fit to data by minimizing the RMS error
    between the data and the fitted sum using `scipy.optimize.minimize`.
    This function attempts to find the optimal set of start fractions (time points at which each exponential
    component begins) that best fit the provided data with a sum of exponentials. Optionally, the time constants
    (taus) for each component can be fixed. The optimization is performed by minimizing the root mean square (RMS)
    of the residuals between the data and the fitted model.
    Parameters
    ----------
    t : np.ndarray
        1D array of time points in nanoseconds, representing the time resolution of the pulse.
    y : np.ndarray
        1D array of amplitude values of the pulse in volts, corresponding to each time point in `t`.
    start_fractions : list of float
        Initial guess for the start fractions (time points) for each exponential component. Must be in descending order.
    bounds_scale : float, optional
        Scale factor for bounds around start fractions during optimization (default is 0.5, meaning ±50%).
    fixed_taus : list of float or None, optional
        If provided, a list of fixed tau values (in nanoseconds) for each exponential component. If set, only amplitudes
        are fitted and taus are constrained. Must have the same length as `start_fractions`.
    a_dc : float or None, optional
        Constant (DC) term. If not provided, the constant term is estimated from the tail of the data.
    verbose : int, optional
    Verbosity (0: silent, 1: summary, 2: detailed) (default 1).
    Returns
    -------
    success : bool
        True if the optimization converged successfully, False otherwise.
    best_fractions : list of float
        The optimized start fractions (time points) for each exponential component.
    best_components : list of tuple (float, float)
        List of (amplitude, tau) tuples for each fitted exponential component.
    best_dc : float
        The fitted or provided constant (DC) term.
    best_rms : float
        The root mean square (RMS) of the residuals for the best fit.
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 101)
    >>> y = 2.0 * np.exp(-t / 10) + 1.0 * np.exp(-t / 30) + 0.1 + 0.05 * np.random.randn(len(t))
    >>> start_fractions = [80, 40]
    >>> success, best_fractions, best_components, best_dc, best_rms = optimize_start_fractions(
    ...     t, y, start_fractions, bounds_scale=0.5, fixed_taus=None, a_dc=None, verbose=False
    ... )
    >>> print("Success:", success)
    >>> print("Best fractions:", best_fractions)
    >>> print("Best components (amp, tau):", best_components)
    >>> print("Best DC:", best_dc)
    >>> print("Best RMS:", best_rms)
    Notes
    -----
    - The function requires that `start_fractions` are in descending order.
    - If `fixed_taus` is provided, it must have the same length as `start_fractions` and all values must be positive.
    - The function uses `sequential_exp_fit` internally to perform the fitting for each set of start fractions.
    """
    # Validate fixed_taus parameter
    if fixed_taus is not None:
        if len(fixed_taus) != len(start_fractions):
            raise ValueError("fixed_taus must have the same length as start_fractions")
        if any(tau <= 0 for tau in fixed_taus):
            raise ValueError("All fixed_taus values must be positive")

    def objective(x):
        """
        Objective function to minimize: RMS between the data and the fitted sum of
        exponentials.
        """
        # Ensure fractions are ordered in descending order
        if not np.all(np.diff(x) < 0):
            return 1e6  # Return large value if constraint is violated

        components, _, residual = sequential_exp_fit(t, y, x, fixed_taus=fixed_taus, a_dc=a_dc, verbose=verbose)
        if len(components) == len(start_fractions):
            current_rms = np.sqrt(np.mean(residual**2))
        else:
            current_rms = 1e6  # Return large value if fitting fails

        return current_rms

    # Define bounds for optimization
    bounds = []
    for start in start_fractions:
        min_val = start * (1 - bounds_scale)
        max_val = start * (1 + bounds_scale)
        bounds.append((min_val, max_val))
    if verbose > 0:
        print("\nOptimizing start_fractions using scipy.optimize.minimize...")
        print(f"Initial values: {[f'{f:.5f}' for f in start_fractions]}")
        print(f"Bounds: ±{bounds_scale * 100}% around initial values")

    # Run optimization
    result = minimize(
        objective,
        x0=start_fractions,
        bounds=bounds,
        method="Nelder-Mead",  # This method works well for non-smooth functions
        options={"disp": True, "maxiter": 200},
    )

    # Get final results
    if result.success:
        best_fractions = result.x
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))
        if verbose > 0:
            print("\nOptimization successful!")
            print(f"Initial fractions: {[f'{f:.5f}' for f in start_fractions]}")
            print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
            if fixed_taus is not None:
                print(f"Fixed taus: {[f'{tau:.3f} ns' for tau in fixed_taus]}")
            print(f"Final RMS: {best_rms:.3e}")
            print(f"Number of iterations: {result.nit}")
    else:
        if verbose > 0:
            print("\nOptimization failed. Using initial values.")
        best_fractions = start_fractions
        components, a_dc, best_residual = sequential_exp_fit(
            t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False
        )
        best_rms = np.sqrt(np.mean(best_residual**2))

    components = [(amp * np.exp(t[0] / tau), tau) for amp, tau in components]
    if verbose > 0:
        print("Optimized components [(a1, tau1), (a2, tau2)...]:")
        print(components)
    return result.success, best_fractions, components, a_dc, best_rms


def plot_fit(t_data: np.ndarray, y_data: np.ndarray, components: List[Tuple[float, float]], a_dc: float):
    """Plot exponential fit results with both linear and log scales.

    Args:
        t_data (np.ndarray): Time points in nanoseconds
        y_data (np.ndarray): Measured flux response data
        components (List[Tuple[float, float]]): List of (amplitude, tau) pairs for each fitted component
        a_dc (float): Constant term

    Returns:
        tuple: (fig, axs) where:
            - fig: Figure object
            - axs: List of axes objects
    """

    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot - linear scale
    axs[0].plot(t_data, y_data, label="Data")
    axs[0].plot(t_data, y_fit, label="Fit")
    axs[0].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[0].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[0].set_xlabel("Time (ns)")
    axs[0].set_ylabel("Flux Response")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Second subplot - log scale
    axs[1].plot(t_data, y_data, label="Data")
    axs[1].plot(t_data, y_fit, label="Fit")
    axs[1].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[1].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[1].set_xlabel("Time (ns)")
    axs[1].set_ylabel("Flux Response")
    axs[1].set_xscale("log")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.tight_layout()

    return fig, axs
