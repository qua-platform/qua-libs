"""Standalone simulation helper functions for gate virtualization analysis tests.

Kept in a separate module (no relative imports) so they can be imported both
via pytest package discovery *and* when running test files directly (e.g.
PyCharm pydevd / ``python test_01_...py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr


def sweep_voltages_mV(
    center_V: float, span_V: float, n_points: int
) -> np.ndarray:
    """Build a sweep array in **mV** from node-parameter-style values (in V).

    This is the bridge between node parameters (volts) and the qarray model
    which expects millivolts.
    """
    half = span_V / 2
    return np.linspace((center_V - half) * 1e3, (center_V + half) * 1e3, n_points)


def simulate_sensor_device_scan(
    model,
    v_sensor: np.ndarray,
    v_device: np.ndarray,
    sensor_gate_idx: int = 6,
    device_gate_idx: int = 0,
    *,
    sensor_operating_point: float = 0.0,
    base_voltages: Optional[np.ndarray] = None,
    compensation_alpha: float = 0.0,
) -> xr.Dataset:
    """Simulate a 2D scan of sensor gate vs device gate using a qarray model.

    Parameters
    ----------
    model : ChargeSensedDotArray
        A configured qarray model.
    v_sensor : np.ndarray
        1-D sensor gate voltages (in mV, length M).
    v_device : np.ndarray
        1-D device gate voltages (in mV, length N).
    sensor_gate_idx : int
        Gate index for the sensor (default 6).
    device_gate_idx : int
        Gate index for the device gate being swept (default 0).
    sensor_operating_point : float
        Constant offset added to the sensor gate voltage.
    base_voltages : np.ndarray, optional
        Base voltage vector (length = n_gates).  Defaults to zeros.
    compensation_alpha : float
        Cross-talk coefficient used to apply virtual gate compensation.
        When non-zero the physical sensor voltage is adjusted by
        ``-alpha * v_device`` at each point, emulating a compensated
        (virtualized) sweep.

    Returns
    -------
    xr.Dataset
        Dataset with ``I`` and ``Q`` variables, dimensions
        ``(sensors, y_volts, x_volts)`` — matching the node's raw format.
        ``x_volts`` = sensor gate, ``y_volts`` = device gate.
    """
    n_gates = sensor_gate_idx + 1  # typically 7
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    rows = []
    for vd in v_device:
        for vs in v_sensor:
            v = base_voltages.copy()
            v[sensor_gate_idx] = sensor_operating_point + vs + compensation_alpha * vd
            v[device_gate_idx] = vd
            rows.append(v)

    voltage_array = np.array(rows)
    z, _ = model.charge_sensor_open(-voltage_array)
    z = z.squeeze()
    signal_2d = z.reshape(len(v_device), len(v_sensor))

    v_sensor_V = v_sensor * 1e-3
    v_device_V = v_device * 1e-3

    I_data = signal_2d[np.newaxis, :, :]
    Q_data = np.zeros_like(I_data)

    ds = xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": ["sensor_1"],
                    "x_volts": v_sensor_V,
                    "y_volts": v_device_V,
                },
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": ["sensor_1"],
                    "x_volts": v_sensor_V,
                    "y_volts": v_device_V,
                },
            ),
        }
    )
    return ds