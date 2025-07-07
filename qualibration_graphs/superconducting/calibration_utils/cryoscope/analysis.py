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

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"

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
    flux_vals = ds_fit.flux.values
    range_vals = final_vals - first_vals
    guess_val = 0.37 * range_vals + first_vals
    idx_closest = np.abs(flux_vals - guess_val.data).argmin()

    if node.parameters.exp_1_tau_guess is not None:
        initial_guess_tau = node.parameters.exp_1_tau_guess
    else:
        initial_guess_tau = ds_fit.time.values[idx_closest]

    print(f"Initial guess for tau: {initial_guess_tau}")

    try:
        p0 = [final_vals, -1 + first_vals / final_vals, initial_guess_tau]
        fit, _ = curve_fit(expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
        fit1_success = True
    except Exception as e:
        fit = p0
        fit1_success = False
        print("single exp fit failed with error:\n", e)
    try:
        p0 = [fit[0], fit[1], 5, fit[1], fit[2]]
        fit2, _ = curve_fit(two_expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit=qubit).values, p0=p0)
        fit2_success = True
    except Exception as e:
        fit2 = None
        fit2_success = False
        print("two exp fit failed with error:\n", e)

    # Save fit results as attributes or DataArrays
    ds_fit.attrs["fit_1exp"] = fit
    ds_fit.attrs["fit_1exp_success"] = fit1_success
    ds_fit.attrs["fit_2exp"] = fit2
    ds_fit.attrs["fit_2exp_success"] = fit2_success

    ds["fit_results"] = ds_fit

    fit, fit_results = _extract_relevant_fit_parameters(ds, node)

    return fit, fit_results


def _extract_relevant_fit_parameters(ds: xr.Dataset, node: QualibrationNode):
    """Extract relevant fit parameters from the dataset and add metadata."""
    # Assess whether the fit was successful or not

    fit = ds["fit_results"]
    fit_results = {
        q: FitParameters(
            fit1_success=fit.attrs.get("fit_1exp_success", False),
            fit2_success=fit.attrs.get("fit_2exp_success", False),
            fit1_A=fit.attrs.get("fit_1exp", [None, None, None])[1],
            fit1_tau=fit.attrs.get("fit_1exp", [None, None, None])[2],
            fit2_A1=fit.attrs.get("fit_2exp", [None, None, None, None, None])[1],
            fit2_tau1=fit.attrs.get("fit_2exp", [None, None, None, None, None])[2],
            fit2_A2=fit.attrs.get("fit_2exp", [None, None, None, None, None])[3],
            fit2_tau2=fit.attrs.get("fit_2exp", [None, None, None, None, None])[4],
        )
        for q in fit.qubit.values
    }
    return ds, fit_results


def log_fitted_results():
    pass


@dataclass
class FitParameters:
    fit1_success: bool = False
    fit2_success: bool = False
    fit1_A: Optional[float] = None
    fit1_tau: Optional[float] = None
    fit2_A1: Optional[float] = None
    fit2_tau1: Optional[float] = None
    fit2_A2: Optional[float] = None
    fit2_tau2: Optional[float] = None
