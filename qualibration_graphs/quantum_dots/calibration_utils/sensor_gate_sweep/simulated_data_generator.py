from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors


def _lorentzian(x, x0, gamma, amplitude, offset):
    """Lorentzian peak: amplitude * (gamma/2)^2 / ((x-x0)^2 + (gamma/2)^2) + offset."""
    return amplitude * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2) + offset


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated Coulomb peak data for the sensor gate sweep.

    For each sensor, a Lorentzian peak is placed at a seeded position
    (not necessarily the centre) within the bias-offset sweep range.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``sensors`` and ``sweep_axes`` into ``node.namespace``.
    """
    node.namespace["sensors"] = sensors = get_sensors(node)

    bias_offsets = np.arange(
        node.parameters.offset_min,
        node.parameters.offset_max,
        node.parameters.offset_step,
    )

    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(
            bias_offsets,
            attrs={"long_name": "Sensor bias offset", "units": "V"},
        ),
    }

    rng = np.random.default_rng(seed=42)

    I_data = []
    Q_data = []

    for i, _sensor in enumerate(sensors):
        sweep_range = node.parameters.offset_max - node.parameters.offset_min
        peak_center = node.parameters.offset_min + rng.uniform(0.25, 0.75) * sweep_range
        peak_gamma = rng.uniform(0.01, 0.05)
        peak_amp = rng.uniform(5e-4, 2e-3)
        baseline = rng.uniform(1e-4, 5e-4)

        signal = _lorentzian(bias_offsets, peak_center, peak_gamma, peak_amp, baseline)

        phase = np.pi / 6 + 0.3 * i
        noise = 2e-5
        I_data.append(
            signal * np.cos(phase) + rng.normal(0, noise, size=len(bias_offsets))
        )
        Q_data.append(
            signal * np.sin(phase) + rng.normal(0, noise, size=len(bias_offsets))
        )

    sensor_names = sensors.get_names()
    coords = {"sensors": sensor_names, "bias_offsets": bias_offsets}
    dims = ["sensors", "bias_offsets"]

    return xr.Dataset(
        {
            "I": xr.DataArray(I_data, dims=dims, coords=coords),
            "Q": xr.DataArray(Q_data, dims=dims, coords=coords),
        }
    )
