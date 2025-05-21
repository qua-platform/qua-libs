import matplotlib.pylab as plt
import numpy as np
from quam.components.quantum_components import qubit
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data.processing import _apply_angle
from scipy.optimize import minimize
from scipy.signal import deconvolve, savgol_filter
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit, minimize




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

    # Generate points for the fitted ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a_fit * np.cos(theta)
    ellipse_y = b_fit * np.sin(theta)

    # Rotate the ellipse points
    cos_angle = np.cos(angle_fit)
    sin_angle = np.sin(angle_fit)
    x_ellipse_rot = cos_angle * ellipse_x - sin_angle * ellipse_y + cx_fit
    y_ellipse_rot = sin_angle * ellipse_x + cos_angle * ellipse_y + cy_fit

    # # Plot the original data points
    # plt.scatter(x, y, label='Data Points')

    # # Plot the fitted ellipse
    # plt.plot(x_ellipse_rot, y_ellipse_rot, color='r', label='Fitted Ellipse')
    # plt.show()

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


def cryoscope_frequency(ds, stable_time_indices, quad_term=-1, sg_range=3, sg_order=2, plot=False):
    ds = ds.copy()

    ds_max = ds.sel(time=slice(stable_time_indices[0], stable_time_indices[1])).max(dim="time")
    ds_min = ds.sel(time=slice(stable_time_indices[0], stable_time_indices[1])).min(dim="time")
    ds_offset = (ds_max + ds_min) / 2
    ds -= ds_offset

    # if plot:
    #     plt.scatter(ds.sel(axis="x"), ds.sel(axis="y"))
    #     plt.gca().set_aspect("equal", adjustable="box")
    #     plt.xlabel("<X>")
    #     plt.ylabel("<Y>")
    #     plt.show()

    angle = _apply_angle(ds.sel(axis="x") + 1j * ds.sel(axis="y"), "time")
    ds["angle"] = angle
    # if plot:
    #     angle.plot()
    #     plt.show()

    freq_cryoscope = diff_savgol(angle, "time", range=sg_range, order=sg_order)
    ds["freq"] = freq_cryoscope
    # if plot:
    #     (-freq_cryoscope).plot()
    #     plt.title("Frequency")
    #     plt.show()
    flux_cryoscope = np.sqrt(np.abs(1e9 * freq_cryoscope / quad_term)).fillna(0)
    # if plot:
    #     flux_cryoscope.plot()
    #     plt.title("Flux")
    #     plt.show()
    if quad_term == -1:
        flux_cryoscope = flux_cryoscope / flux_cryoscope.sel(time=slice(stable_time_indices[0], stable_time_indices[1])).mean(dim="time")

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

    sg_order = 2
    sg_range = 3

    ds_fit = cryoscope_frequency(
        ds[data],
        quad_term=-1,
        stable_time_indices=(node.parameters.cryoscope_len - 20, node.parameters.cryoscope_len),
        sg_order=sg_order,
        sg_range=sg_range,
    )

    first_vals = ds_fit.flux.sel(time=slice(0, 1)).mean()
    final_vals = ds_fit.flux.sel(time=slice(node.parameters.cryoscope_len - 20, None)).mean()

    try:
        p0 = [final_vals, -1 + first_vals / final_vals, 50]
        fit, _ = curve_fit(expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit="qD1").values, p0=p0)
    except:
        fit = p0
        print("single exp fit failed")
    try:
        p0 = [fit[0], fit[1], 5, fit[1], fit[2]]
        fit2, _ = curve_fit(two_expdecay, ds_fit.time.values, ds_fit.flux.sel(qubit="qD1").values, p0=p0)
    except:
        fit2 = None
        print("two exp fit failed")

    # Save fit results as attributes or DataArrays
    ds_fit.attrs["fit_1exp"] = fit
    ds_fit.attrs["fit_2exp"] = fit2

    ds["fit_results"] = ds_fit

    return ds


def log_fitted_results():
    pass
