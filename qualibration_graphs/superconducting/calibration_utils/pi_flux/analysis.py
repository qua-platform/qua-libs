from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr
from numpy.polynomial import Polynomial as P
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit, minimize


@dataclass
class PiFluxParameters:
    fit_successful: bool
    optimized_fractions: List[float]
    a_tau_tuple: List[Tuple[float, float]]
    a_dc: float
    rms_error: float


def log_fitted_results(fit_results: Dict[str, PiFluxParameters], log_callable=print) -> None:
    for qb, res in fit_results.items():
        if res.fit_successful:
            log_callable(f"{qb}: SUCCESS, a_dc={res.a_dc:.3e}, rms={res.rms_error:.3e}, comps={res.a_tau_tuple}")
        else:
            log_callable(f"{qb}: FAILED")


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V, add amplitude/phase, add freq_full if detuning present."""
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])  # type: ignore
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    if "detuning" in ds.coords:
        full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
        ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
        ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian(freqs, states):
    p0 = [
        float(np.max(states) - np.min(states)),
        float(freqs[np.argmax(states)]),
        float((freqs[-1] - freqs[0]) / 10),
        float(np.min(states)),
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return float(popt[1])
    except Exception:
        return float("nan")


def extract_center_freqs_state(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    """Extract center frequencies from state discrimination dataset as a DataArray.

    Ensures we operate on the 'state' DataArray (not the whole Dataset) so apply_ufunc
    returns a DataArray, avoiding assignment errors downstream.
    """
    freq_dim = "detuning" if "detuning" in ds.dims else "freq"
    state_da = ds["state"].transpose("qubit", "time", freq_dim)
    center_freqs = xr.apply_ufunc(
        lambda states: fit_gaussian(freqs, states),
        state_da,
        input_core_dims=[[freq_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs.rename("center_frequency")


def extract_center_freqs_iq(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    iq_data = ds["IQ_abs"] if "IQ_abs" in ds.data_vars else ds.get("I", None)
    if iq_data is None:
        raise ValueError("Dataset is missing IQ_abs and I data variables for IQ analysis")
    stacked = iq_data.transpose("qubit", "time", "detuning")
    center_freqs = xr.apply_ufunc(
        lambda iq_slice: fit_gaussian(freqs, iq_slice),
        stacked,
        input_core_dims=[["detuning"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs


def single_exp_decay(t, amp, tau):
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


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, PiFluxParameters]]:
    """Compute center_freqs, detuning/freq_full/flux coords, flux_response, and fit cascade."""
    qubits = node.namespace["qubits"]

    # Frequency points array
    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else (ds.get("detuning").values if "detuning" in ds.dims else ds.get("freq").values)
    )

    # Ensure detuning naming aligns
    if "detuning" not in ds.dims and "freq" in ds.dims:
        ds = ds.rename({"freq": "detuning"})

    # Center frequencies from either state or IQ
    if node.parameters.use_state_discrimination and "state" in ds.data_vars:
        center_freqs = extract_center_freqs_state(ds, dfs)
    else:
        center_freqs = extract_center_freqs_iq(ds, dfs)

    # Add example-style coords
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency - node.parameters.detuning_in_mhz * 1e6 for q in qubits]),
            ),
            "detuning": (
                ["qubit", "detuning"],
                np.array([dfs - node.parameters.detuning_in_mhz * 1e6 for q in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.array(
                    [
                        (
                            np.sqrt(np.maximum(0.0, dfs / node.parameters.detuning_in_mhz * 1e6))
                            if q.freq_vs_flux_01_quad_term != 0
                            else np.full_like(dfs, np.nan, dtype=float)
                        )
                        for q in qubits
                    ]
                ),
            ),
        }
    )

    # Add flux-induced static shift as in example
    times = ds["time"].values
    center_freqs = center_freqs + np.array(
        [-node.parameters.detuning_in_mhz * 1e6 * np.ones_like(times) for q in qubits]
    )

    quad_terms = xr.DataArray(
        [q.freq_vs_flux_01_quad_term for q in qubits], coords={"qubit": [q.name for q in qubits]}, dims=["qubit"]
    )
    flux_response = xr.where(quad_terms != 0, np.sqrt(center_freqs / quad_terms), np.nan)

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response

    fit_results: Dict[str, PiFluxParameters] = {}
    for q in qubits:
        t_data = flux_response.sel(qubit=q.name).time.values
        y_data = flux_response.sel(qubit=q.name).values
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
            t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
        )
        fit_results[q.name] = PiFluxParameters(
            fit_successful=fit_successful,
            optimized_fractions=best_fractions,
            a_tau_tuple=best_components,
            a_dc=best_a_dc,
            rms_error=best_rms,
        )
    return ds, fit_results


def decompose_exp_sum_to_cascade(
    A: Sequence, tau: Sequence, A_dc: float = 1.0, compensate_v34_fpga_scale: bool = True, Ts: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
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

    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))

    assert np.all(np.isreal(zeros)) and np.all(
        np.isreal(poles)
    ), "Got complex zeros; this configuration can't be inverted or decomposed to cascade of single pole stages"

    tau_c = -1 / poles
    A_c = poles / zeros - 1

    scale = 1 / A_dc

    if compensate_v34_fpga_scale:
        scale *= get_scaling_of_v34_fpga_filter(A_c, tau_c, Ts)

    return A_c, tau_c, scale


def get_scaling_of_v34_fpga_filter(A_c: np.ndarray, tau_c: np.ndarray, Ts) -> float:
    """get_scaling_of_v34_fpga_filter
    Calculate the scaling factor for the V3.4 FPGA filter implementation.
    This scaling is necessary to make the cascade of single exponent filters equivalent to the sum of exponents.
    :param A_c: Amplitudes of the cascade filters
    :param tau_c: Time constants of the cascade filters
    :param Ts: Sampling period
    :return: scale
    """
    return float(np.prod((Ts + 2 * tau_c) / (Ts + 2 * tau_c * (1 + A_c))))


def get_rational_filter_single_exp_cont_time(A: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1, 1 / tau])
    b = np.array([A])
    return b, a


# The functions below are used to decompose the sum of exponentials to a cascade of single
# exponent filters, as implemented in QOP 3.4.
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
