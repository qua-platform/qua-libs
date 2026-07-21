from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from qualang_tools.units import unit

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate a synthetic frequency x detuning dataset with a localized PCA feature."""
    u = unit(coerce_to_integer=True)

    node.namespace["sensors"] = sensors = get_sensors(node)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    frequency = np.arange(-span / 2, +span / 2, step)
    detuning = np.arange(
        node.parameters.detuning_start,
        node.parameters.detuning_stop,
        node.parameters.detuning_step,
    )

    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "frequency": xr.DataArray(
            frequency,
            attrs={"long_name": "readout frequency", "units": "Hz"},
        ),
        "detuning": xr.DataArray(
            detuning,
            attrs={"long_name": "quantum dot pair detuning", "units": "V"},
        ),
    }

    rng = np.random.default_rng(seed=42)

    I_data = []
    Q_data = []
    freq_grid, det_grid = np.meshgrid(frequency, detuning, indexing="ij")

    for sensor_index, _sensor in enumerate(sensors):
        center_frequency = rng.uniform(-0.15 * span, 0.15 * span)
        center_detuning = rng.uniform(
            detuning.min() + 0.2 * (detuning.max() - detuning.min()),
            detuning.max() - 0.2 * (detuning.max() - detuning.min()),
        )
        sigma_frequency = rng.uniform(0.04 * span, 0.08 * span)
        sigma_detuning = rng.uniform(
            0.08 * (detuning.max() - detuning.min()),
            0.16 * (detuning.max() - detuning.min()),
        )

        transition = np.exp(
            -0.5
            * (
                ((freq_grid - center_frequency) / sigma_frequency) ** 2
                + ((det_grid - center_detuning) / sigma_detuning) ** 2
            )
        )
        background = (
            0.1e-3 * (freq_grid / max(np.max(np.abs(frequency)), 1.0))
            + 0.1e-3 * (det_grid - np.mean(detuning))
        )
        phase = rng.uniform(0, 2 * np.pi)
        signal = (0.8e-3 * transition + background) * np.exp(1j * phase)
        signal += 0.05e-3 * transition * sensor_index

        noise_scale = 0.04e-3
        I_data.append(signal.real + rng.normal(0.0, noise_scale, size=signal.shape))
        Q_data.append(signal.imag + rng.normal(0.0, noise_scale, size=signal.shape))

    sensor_names = sensors.get_names()
    coords = {"sensor": sensor_names, "frequency": frequency, "detuning": detuning}
    dims = ["sensor", "frequency", "detuning"]

    return xr.Dataset(
        {
            "I": xr.DataArray(I_data, dims=dims, coords=coords),
            "Q": xr.DataArray(Q_data, dims=dims, coords=coords),
        }
    )
