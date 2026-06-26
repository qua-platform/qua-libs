from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors

SIMULATED_DELAY_NS = 224


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated ADC traces for LF-FEM time-of-flight calibration.

    Simulates the raw ADC capture of a readout pulse (a cosine burst at
    the resonator's intermediate frequency) arriving after a propagation
    delay.  Before the pulse arrives, the trace contains only noise.

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

    # Raw ADC counts; process_raw_dataset converts via  volts = -counts / 2^12.
    pulse_amplitude_counts = 0.1 * 4096
    noise_std = 5.0

    adc_data = []
    adc_single_data = []

    for i, sensor in enumerate(sensors):
        delay = SIMULATED_DELAY_NS + 4 * i

        if_freq = sensor.readout_resonator.intermediate_frequency
        omega = 2 * np.pi * if_freq * 1e-9  # rad/ns

        # Cosine burst: zero before delay, oscillation after
        waveform = np.zeros_like(time_axis, dtype=float)
        pulse_mask = time_axis >= delay
        waveform[pulse_mask] = pulse_amplitude_counts * np.cos(
            omega * (time_axis[pulse_mask] - delay)
        )

        trace_avg = waveform + rng.normal(0, noise_std * 0.3, size=len(time_axis))
        trace_single = waveform + rng.normal(0, noise_std, size=len(time_axis))

        adc_data.append(trace_avg)
        adc_single_data.append(trace_single)

    sensor_names = sensors.get_names()
    coords = {"sensor": sensor_names, "readout_time": time_axis}
    dims = ["sensor", "readout_time"]

    return xr.Dataset(
        {
            "adc": xr.DataArray(adc_data, dims=dims, coords=coords),
            "adc_single_run": xr.DataArray(adc_single_data, dims=dims, coords=coords),
        }
    )
