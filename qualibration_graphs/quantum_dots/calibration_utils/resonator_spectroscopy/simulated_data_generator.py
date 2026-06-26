from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from qualang_tools.units import unit

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated 1D resonator spectroscopy data.

    Models the complex reflection coefficient S11 of a notch-type
    resonator: far from resonance the signal is close to the baseline;
    on resonance there is both an amplitude dip and a ~pi phase swing.
    Seeded standing-wave ripples are superimposed on the baseline to
    mimic cable reflections.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``sensors`` and ``sweep_axes`` into ``node.namespace``.
    """
    u = unit(coerce_to_integer=True)

    node.namespace["sensors"] = sensors = get_sensors(node)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "detuning": xr.DataArray(
            dfs,
            attrs={"long_name": "readout frequency", "units": "Hz"},
        ),
    }

    rng = np.random.default_rng(seed=42)

    I_data = []
    Q_data = []

    for i, _sensor in enumerate(sensors):
        dip_shift = rng.uniform(-span * 0.05, span * 0.05)
        kappa = rng.uniform(0.5e6, 2e6)
        coupling = 0.7

        # Complex S11 of a notch resonator:
        #   S11(f) = 1 - coupling * kappa / (kappa/2 + j*(f - f0))
        # This gives an amplitude dip AND a phase swing through resonance.
        delta = dfs - dip_shift
        denom = kappa / 2 + 1j * delta
        s11 = 1.0 - coupling * (kappa / 2) / denom

        baseline = 1e-3 * (1 + 0.1 * i)

        # Standing-wave ripples on the baseline (cable reflections)
        ripple = np.zeros_like(dfs)
        n_modes = rng.integers(2, 5)
        for _ in range(n_modes):
            period = rng.uniform(span * 0.1, span * 0.4)
            ripple_amp = rng.uniform(0.02, 0.08)
            ripple_phase = rng.uniform(0, 2 * np.pi)
            ripple += ripple_amp * np.sin(2 * np.pi * dfs / period + ripple_phase)

        # Overall complex signal: baseline * (1 + ripple) * S11, rotated by
        # an arbitrary cable phase.
        cable_phase = rng.uniform(0, 2 * np.pi)
        signal = baseline * (1 + ripple) * s11 * np.exp(1j * cable_phase)

        noise = 1e-5
        I_data.append(signal.real + rng.normal(0, noise, size=len(dfs)))
        Q_data.append(signal.imag + rng.normal(0, noise, size=len(dfs)))

    sensor_names = sensors.get_names()
    coords = {"sensors": sensor_names, "detuning": dfs}
    dims = ["sensors", "detuning"]

    return xr.Dataset(
        {
            "I": xr.DataArray(I_data, dims=dims, coords=coords),
            "Q": xr.DataArray(Q_data, dims=dims, coords=coords),
        }
    )
