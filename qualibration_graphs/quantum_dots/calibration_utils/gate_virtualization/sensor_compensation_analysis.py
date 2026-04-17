"""Analysis functions for sensor gate compensation (node 01).

Extracts the cross-talk coefficient between sensor gates and device gates
from 2D scans by fitting a shifted Lorentzian model.  The peak position
shifts linearly with device gate voltage; the slope of that shift is the
virtual gate term (alpha).

Two fitting methods are available:

1. **Global shifted Lorentzian** (default / legacy): fits a single linear
   peak trajectory ``v0 + alpha * v_d`` via ``differential_evolution``.
   Fast but biased when charge transitions shift the peak abruptly.

2. **Bayesian piecewise-linear Lorentzian** (``method="bayesian_cp"``):
   extracts peak positions per row, then finds alpha by maximising the
   BayesianCP log-evidence on the detrended trace ``peaks - alpha * v_d``.
   This jointly determines the gradient and change points without a
   biased initial estimate.  Robust to charge-transition outliers.

Model
-----
    f(v_s, v_d) = A / (1 + ((v_s - center(v_d)) / gamma)^2) + offset

    center(v_d) = alpha * v_d + c_k     (segment k between change points)

The compensation matrix entry is ``-alpha``: sweeping the device gate by
``dV`` requires compensating the sensor gate by ``-alpha * dV``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

import xarray as xr


def shifted_lorentzian_2d(
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    A: float,
    v0: float,
    alpha: float,
    gamma: float,
    offset: float,
) -> np.ndarray:
    """Evaluate the shifted Lorentzian on a 2D grid.

    Parameters
    ----------
    v_sensor : np.ndarray
        1-D sensor gate voltages (length M).
    v_device : np.ndarray
        1-D device gate voltages (length N).
    A, v0, alpha, gamma, offset : float
        Model parameters.

    Returns
    -------
    np.ndarray
        Shape ``(N, M)`` — signal[i, j] corresponds to
        ``(v_device[i], v_sensor[j])``.
    """
    vs = v_sensor[np.newaxis, :]  # (1, M)
    vd = v_device[:, np.newaxis]  # (N, 1)
    peak_pos = v0 + alpha * vd
    return A / (1.0 + ((vs - peak_pos) / gamma) ** 2) + offset


def fit_shifted_lorentzian(
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    signal: np.ndarray,
    bounds: Optional[list[Tuple[float, float]]] = None,
    *,
    seed: int = 0,
    maxiter: int = 1000,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Fit a shifted Lorentzian to a 2D scan using differential evolution.

    Parameters
    ----------
    v_sensor : np.ndarray
        1-D sensor gate voltages (length M).
    v_device : np.ndarray
        1-D device gate voltages (length N).
    signal : np.ndarray
        2-D measured signal, shape ``(N, M)`` — rows indexed by *v_device*,
        columns by *v_sensor*.
    bounds : list of (min, max) tuples, optional
        Bounds for ``[A, v0, alpha, gamma, offset]``.  Estimated
        automatically when *None*.
    seed : int
        RNG seed for reproducibility.
    maxiter : int
        Maximum generations for differential evolution.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        ``{"A", "v0", "alpha", "gamma", "offset", "residual", "success"}``.
    """
    signal = np.asarray(signal, dtype=float)
    v_sensor = np.asarray(v_sensor, dtype=float)
    v_device = np.asarray(v_device, dtype=float)

    if bounds is None:
        bounds = _estimate_bounds(v_sensor, v_device, signal)

    def cost(params: np.ndarray) -> float:
        model = shifted_lorentzian_2d(v_sensor, v_device, *params)
        return float(np.sum((signal - model) ** 2))

    result = differential_evolution(
        cost,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=tol,
    )

    A, v0, alpha, gamma, offset = result.x
    return {
        "A": float(A),
        "v0": float(v0),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "offset": float(offset),
        "residual": float(result.fun),
        "success": bool(result.success),
    }


def _estimate_bounds(
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    signal: np.ndarray,
) -> list[Tuple[float, float]]:
    """Heuristic parameter bounds from the data ranges and signal statistics."""
    sig_min, sig_max = float(np.min(signal)), float(np.max(signal))
    sig_range = sig_max - sig_min if sig_max > sig_min else 1.0

    vs_min, vs_max = float(v_sensor.min()), float(v_sensor.max())
    vs_span = vs_max - vs_min if vs_max > vs_min else 1.0
    vd_span = float(v_device.max()) - float(v_device.min())
    vd_span = vd_span if vd_span > 0 else 1.0

    # A: amplitude can be positive or negative
    A_bound = (-2.0 * sig_range, 2.0 * sig_range)
    # v0: peak centre should be within the sensor range (with margin)
    v0_bound = (vs_min - vs_span, vs_max + vs_span)
    # alpha: slope magnitude bounded by sensor span / device span
    alpha_max = 3.0 * vs_span / vd_span
    alpha_bound = (-alpha_max, alpha_max)
    # gamma: HWHM between a tiny fraction and the full sensor span
    gamma_bound = (vs_span * 1e-4, vs_span * 2.0)
    # offset: around the signal baseline
    offset_bound = (sig_min - sig_range, sig_max + sig_range)

    return [A_bound, v0_bound, alpha_bound, gamma_bound, offset_bound]


def fit_shifted_lorentzian_bayesian_cp(
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    signal: np.ndarray,
    *,
    hazard: float = 1 / 30.0,
    cp_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Fit a shifted Lorentzian robust to charge transitions via BayesianCP.

    Pipeline:
    1. Estimate shared Lorentzian shape (A, gamma, offset) from the data.
    2. Fit the peak center per row with shape held fixed.
    3. Find alpha by maximising BCP log-evidence on ``peaks - alpha * v_d``,
       jointly determining the gradient and change points.
    4. Final BCP run at optimal alpha; per-segment intercepts via lstsq.

    Parameters
    ----------
    v_sensor, v_device, signal
        Same as :func:`fit_shifted_lorentzian`.
    hazard : float
        Hazard rate for BayesianCP (smaller → fewer expected change points).
    cp_threshold : float
        Posterior probability above which a point is classified as a
        change point.

    Returns
    -------
    dict
        Same keys as :func:`fit_shifted_lorentzian` plus
        ``"changepoint_indices"``, ``"v0_segments"``, ``"n_changepoints"``,
        ``"alpha_std"``, ``"peak_positions"``, and ``"cp_posterior"``.
    """
    from calibration_utils.bayesian_change_point.shifted_lorentzian_cp import (
        fit_robust_sensor_compensation,
    )

    result = fit_robust_sensor_compensation(
        v_sensor,
        v_device,
        signal,
        hazard=hazard,
        cp_threshold=cp_threshold,
    )

    if not result.segment_intercepts:
        raise RuntimeError("BCP fit produced no segment intercepts — unable to determine v0.")
    v0_effective = result.segment_intercepts[0]

    return {
        "A": result.A,
        "v0": v0_effective,
        "alpha": result.alpha,
        "gamma": result.gamma,
        "offset": result.offset,
        "residual": result.alpha_std,
        "success": True,
        "alpha_std": result.alpha_std,
        "n_changepoints": result.n_changepoints,
        "changepoint_indices": result.changepoint_indices,
        "v0_segments": result.segment_intercepts,
        "peak_positions": result.peak_positions,
        "cp_posterior": result.cp_posterior,
    }


def _extract_axes(
    ds: xr.Dataset,
    signal_var: str,
    sensor_gate_name: str,
    device_gate_name: str,
    sensor_idx: int,
    sensor_axis_name: str,
    device_axis_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Extract v_sensor, v_device, signal arrays and resolved coord names."""
    data = ds[signal_var]

    if "sensors" in data.dims:
        data = data.isel(sensors=sensor_idx)

    if sensor_gate_name in data.coords and device_gate_name in data.coords:
        sensor_coord_name = sensor_gate_name
        device_coord_name = device_gate_name
    else:
        sensor_coord_name = sensor_axis_name
        device_coord_name = device_axis_name

    missing = [coord for coord in (sensor_coord_name, device_coord_name) if coord not in data.coords]
    if missing:
        raise KeyError(f"Missing coordinate(s) {missing} in dataset. " f"Available coords: {list(data.coords)}")

    v_sensor = data.coords[sensor_coord_name].values
    v_device = data.coords[device_coord_name].values

    dim_order = list(data.dims)
    dev_idx = dim_order.index(device_coord_name)
    sen_idx = dim_order.index(sensor_coord_name)
    if dev_idx > sen_idx:
        signal = data.values.T
    else:
        signal = data.values

    return v_sensor, v_device, signal, sensor_coord_name, device_coord_name


def extract_sensor_compensation_coefficients(
    ds: xr.Dataset,
    sensor_gate_name: str,
    device_gate_name: str,
    *,
    signal_var: str = "amplitude",
    sensor_idx: int = 0,
    sensor_axis_name: str = "x_volts",
    device_axis_name: str = "y_volts",
    method: str = "global",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract the cross-talk coefficient from a processed 2D scan dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset containing *signal_var* with dimensions
        ``(sensors, <device_gate_dim>, <sensor_gate_dim>)`` or
        ``(<device_gate_dim>, <sensor_gate_dim>)``.
    sensor_gate_name : str
        Name of the sensor gate on the X scan axis.
    device_gate_name : str
        Name of the device gate on the Y scan axis.
    signal_var : str
        Name of the data variable to fit (default ``"amplitude"``).
    sensor_idx : int
        Index along the ``sensors`` dimension if present.
    sensor_axis_name : str
        Coordinate/dimension name of the sensor axis in *ds*
        (default ``"x_volts"``).
    device_axis_name : str
        Coordinate/dimension name of the device axis in *ds*
        (default ``"y_volts"``).
    method : str
        ``"global"`` for the legacy single shifted-Lorentzian fit, or
        ``"bayesian_cp"`` for the Bayesian piecewise-linear model that
        is robust to charge transitions.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen fitting function.

    Returns
    -------
    dict
        ``{"coefficient": float, "fit_quality": float, "fit_params": dict}``.
        ``coefficient`` is ``alpha`` (the virtual gate term).
    """
    v_sensor, v_device, signal, sensor_coord, device_coord = _extract_axes(
        ds,
        signal_var,
        sensor_gate_name,
        device_gate_name,
        sensor_idx,
        sensor_axis_name,
        device_axis_name,
    )

    kwargs = dict(fit_kwargs or {})

    if method == "bayesian_cp":
        fit_result = fit_shifted_lorentzian_bayesian_cp(v_sensor, v_device, signal, **kwargs)
    elif method == "global":
        fit_result = fit_shifted_lorentzian(v_sensor, v_device, signal, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'global' or 'bayesian_cp'.")

    return {
        "coefficient": fit_result["alpha"],
        "fit_quality": fit_result["residual"],
        "sensor_gate_name": sensor_gate_name,
        "device_gate_name": device_gate_name,
        "sensor_axis_name": sensor_coord,
        "device_axis_name": device_coord,
        "fit_params": fit_result,
    }
