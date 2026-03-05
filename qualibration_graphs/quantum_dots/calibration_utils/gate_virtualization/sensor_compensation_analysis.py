"""Analysis functions for sensor gate compensation (node 01).

Extracts the cross-talk coefficient between sensor gates and device gates
from 2D scans by fitting a shifted Lorentzian model.  The peak position
shifts linearly with device gate voltage; the slope of that shift is the
virtual gate term (alpha).

Model
-----
    f(v_s, v_d) = A / (1 + ((v_s - (v0 + alpha * v_d)) / gamma)^2) + offset

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


def extract_sensor_compensation_coefficients(
    ds: xr.Dataset,
    sensor_gate_name: str,
    device_gate_name: str,
    *,
    signal_var: str = "amplitude",
    sensor_idx: int = 0,
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
        Coordinate name for the sensor gate axis.
    device_gate_name : str
        Coordinate name for the device gate axis.
    signal_var : str
        Name of the data variable to fit (default ``"amplitude"``).
    sensor_idx : int
        Index along the ``sensors`` dimension if present.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`fit_shifted_lorentzian`.

    Returns
    -------
    dict
        ``{"coefficient": float, "fit_quality": float, "fit_params": dict}``.
        ``coefficient`` is ``alpha`` (the virtual gate term).
    """
    data = ds[signal_var]

    if "sensors" in data.dims:
        data = data.isel(sensors=sensor_idx)

    v_sensor = data.coords[sensor_gate_name].values
    v_device = data.coords[device_gate_name].values

    # Ensure shape is (device, sensor)
    dim_order = list(data.dims)
    dev_idx = dim_order.index(device_gate_name)
    sen_idx = dim_order.index(sensor_gate_name)
    if dev_idx > sen_idx:
        signal = data.values.T
    else:
        signal = data.values

    kwargs = dict(fit_kwargs or {})
    fit_result = fit_shifted_lorentzian(v_sensor, v_device, signal, **kwargs)

    return {
        "coefficient": fit_result["alpha"],
        "fit_quality": fit_result["residual"],
        "fit_params": fit_result,
    }
