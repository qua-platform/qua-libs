from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from qualang_tools.units import unit

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_TAU_NS = 320.0  # ~500 kHz — fallback when estimated_bias_tee_tau_ns is None
# ─────────────────────────────────────────────────────────────────────────


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated IQ data for the frequency-sweep bias tee characterization.

    Generates uncorrected high-pass transfer function data using the
    estimated (or default) time constant.  The correction overlay is
    computed later in analysis from the *extracted* fit parameters.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        The function writes ``elements``, ``sensors``, ``frequencies``, and
        ``sweep_axes`` into ``node.namespace``.
    """
    u = unit(coerce_to_integer=True)

    node.namespace["elements"] = elements = [
        node.machine.get_component(el) for el in node.parameters.elements
    ]
    node.namespace["sensors"] = sensors = get_sensors(node)

    f_start = node.parameters.square_wave_frequency_start_MHz * u.MHz
    f_stop = node.parameters.square_wave_frequency_stop_MHz * u.MHz
    df = node.parameters.square_wave_frequency_step_MHz * u.MHz

    node.namespace["frequencies"] = frequencies = np.arange(f_start, f_stop, df)

    node.namespace["sweep_axes"] = {
        "frequency": xr.DataArray(
            frequencies,
            attrs={"long_name": "frequency", "units": "Hz"},
        ),
    }

    rng = np.random.default_rng(seed=42)
    tau_sim = node.parameters.estimated_bias_tee_tau_ns or DEFAULT_TAU_NS
    f_c_sim = 1e9 / (2 * np.pi * tau_sim)
    noise_level = 2e-5

    data_vars = {}
    for el_idx, el in enumerate(elements):
        for i, sensor in enumerate(sensors):
            scale = 1.0 + 0.05 * el_idx
            A_sim = 1e-3 * scale
            B_sim = 5e-4 * scale

            signal = A_sim * frequencies / np.sqrt(frequencies**2 + f_c_sim**2) + B_sim

            I_vals = signal + rng.normal(0, noise_level, size=len(frequencies))
            Q_vals = signal * 0.3 + rng.normal(0, noise_level, size=len(frequencies))

            suffix = f"{el.name}_{i + 1}"
            data_vars[f"I_{suffix}"] = xr.DataArray(I_vals, dims=["frequency"])
            data_vars[f"Q_{suffix}"] = xr.DataArray(Q_vals, dims=["frequency"])

    return xr.Dataset(data_vars, coords={"frequency": frequencies})
