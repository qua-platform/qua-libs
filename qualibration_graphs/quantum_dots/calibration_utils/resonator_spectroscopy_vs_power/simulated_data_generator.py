from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from qualang_tools.units import unit

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated resonator spectroscopy vs power 2D data.

    For each sensor, the complex S11 of a notch-type resonator is
    simulated across a frequency x power grid. The resonance frequency in
    detuning coordinates does not vary with drive power (``delta`` omits a
    power dependence); linewidth broadens with power via ``kappa``. The notch
    parameters ``kappa_base`` and ``dip_center`` are sampled once for all
    sensors so seeded fake data does not give one unrelated Lorentzian
    draw per channel (which made the derivative-based optimal-power sweep
    succeed for one sensor and arbitrarily fail for another).

    Channels still differ by cable phase and (optional) amplitude scale.


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

    power_dbm = np.linspace(
        node.parameters.min_power_dbm,
        node.parameters.max_power_dbm,
        node.parameters.num_power_points,
    )

    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "detuning": xr.DataArray(
            dfs,
            attrs={"long_name": "readout frequency", "units": "Hz"},
        ),
        "power": xr.DataArray(
            power_dbm,
            attrs={"long_name": "readout power", "units": "dBm"},
        ),
    }

    rng = np.random.default_rng(seed=42)

    # One Lorentzian notch for every sensor; same fixed center vs power (see ``delta``).
    kappa_base = rng.uniform(0.5e6, 1.5e6)
    dip_center = rng.uniform(-span * 0.05, span * 0.05)
    coupling = 0.7

    I_data = []
    Q_data = []

    for i, _sensor in enumerate(sensors):

        # Voltage scales as 10^(P/20) relative to max power
        power_linear = 10 ** ((power_dbm - power_dbm.max()) / 20)
        power_norm = (power_dbm - power_dbm.min()) / (power_dbm.max() - power_dbm.min())

        freq_grid, _ = np.meshgrid(dfs, power_dbm, indexing="ij")
        pwr_lin_grid = np.meshgrid(dfs, power_linear, indexing="ij")[1]
        pwr_norm_grid = np.meshgrid(dfs, power_norm, indexing="ij")[1]

        # Fixed resonance frequency; linewidth broadens slightly at high power
        kappa = kappa_base * (1 + 0.5 * pwr_norm_grid)
        delta = freq_grid - dip_center

        # Complex S11: notch resonator model
        denom = kappa / 2 + 1j * delta
        s11 = 1.0 - coupling * (kappa / 2) / denom

        baseline = 1e-3 * pwr_lin_grid * (1 + 0.1 * i)

        cable_phase = rng.uniform(0, 2 * np.pi)
        signal = baseline * s11 * np.exp(1j * cable_phase)

        noise = 1e-5
        I_data.append(signal.real + rng.normal(0, noise, size=signal.shape))
        Q_data.append(signal.imag + rng.normal(0, noise, size=signal.shape))

    sensor_names = sensors.get_names()
    coords = {"sensor": sensor_names, "detuning": dfs, "power": power_dbm}
    dims = ["sensor", "detuning", "power"]

    return xr.Dataset(
        {
            "I": xr.DataArray(I_data, dims=dims, coords=coords),
            "Q": xr.DataArray(Q_data, dims=dims, coords=coords),
        }
    )
