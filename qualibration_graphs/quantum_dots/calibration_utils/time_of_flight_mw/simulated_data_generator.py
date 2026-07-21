from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors

SIMULATED_DELAY_NS = 224


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated IQ ADC traces for MW-FEM time-of-flight calibration.

    The MW-FEM digitises the reflected readout pulse into I and Q
    quadratures.  This simulates a cosine burst at the resonator IF
    arriving after a propagation delay, split into its I (cos) and Q
    (sin) components.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``sensors`` and ``sweep_axes`` into ``node.namespace``.
    """
    node.namespace["sensors"] = sensors = get_sensors(node)

    readout_length = node.parameters.readout_length_in_ns or 1000
    time_axis = np.arange(0, readout_length, 1)

    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "readout_time": xr.DataArray(
            time_axis,
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    rng = np.random.default_rng(seed=42)

    pulse_amplitude_counts = 0.08 * 4096
    noise_std = 5.0

    adcI_avg, adcQ_avg = [], []
    adcI_single, adcQ_single = [], []

    for i, sensor in enumerate(sensors):
        delay = SIMULATED_DELAY_NS + 4 * i

        if_freq = sensor.readout_resonator.intermediate_frequency
        omega = 2 * np.pi * if_freq * 1e-9  # rad/ns
        reflection_phase = 0.3 * i

        pulse_mask = time_axis >= delay
        t_shifted = time_axis[pulse_mask] - delay

        waveI = np.zeros_like(time_axis, dtype=float)
        waveQ = np.zeros_like(time_axis, dtype=float)
        waveI[pulse_mask] = pulse_amplitude_counts * np.cos(
            omega * t_shifted + reflection_phase
        )
        waveQ[pulse_mask] = pulse_amplitude_counts * np.sin(
            omega * t_shifted + reflection_phase
        )

        adcI_avg.append(waveI + rng.normal(0, noise_std * 0.3, size=len(time_axis)))
        adcQ_avg.append(waveQ + rng.normal(0, noise_std * 0.3, size=len(time_axis)))
        adcI_single.append(waveI + rng.normal(0, noise_std, size=len(time_axis)))
        adcQ_single.append(waveQ + rng.normal(0, noise_std, size=len(time_axis)))

    sensor_names = sensors.get_names()
    coords = {"sensor": sensor_names, "readout_time": time_axis}
    dims = ["sensor", "readout_time"]

    return xr.Dataset(
        {
            "adcI": xr.DataArray(adcI_avg, dims=dims, coords=coords),
            "adcQ": xr.DataArray(adcQ_avg, dims=dims, coords=coords),
            "adc_single_runI": xr.DataArray(adcI_single, dims=dims, coords=coords),
            "adc_single_runQ": xr.DataArray(adcQ_single, dims=dims, coords=coords),
        }
    )
