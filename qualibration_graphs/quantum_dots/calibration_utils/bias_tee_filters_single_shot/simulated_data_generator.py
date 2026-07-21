from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import get_sensors

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_TAU_NS = 20_000.0  # Fallback when estimated_bias_tee_tau_ns is None
# ─────────────────────────────────────────────────────────────────────────


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated IQ data for the single-shot bias tee characterization.

    Generates uncorrected exponential decay data using the estimated (or
    default) time constant.  The correction overlay is computed later in
    analysis from the *extracted* fit parameters.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        The function writes ``elements``, ``sensors``, and ``sweep_axes``
        into ``node.namespace``.
    """
    node.namespace["elements"] = elements = [
        node.machine.get_component(el) for el in node.parameters.elements
    ]
    node.namespace["sensors"] = sensors = get_sensors(node)

    num_chunks = node.parameters.measurement_time // node.parameters.integration_time
    time_array = (np.arange(num_chunks) + 0.5) * node.parameters.integration_time

    node.namespace["sweep_axes"] = {
        "time_array": xr.DataArray(
            time_array,
            attrs={"long_name": "time", "units": "ns"},
        ),
    }

    rng = np.random.default_rng(seed=42)
    tau_sim = node.parameters.estimated_bias_tee_tau_ns or DEFAULT_TAU_NS
    noise_level = 1e-5

    data_vars = {}
    for el_idx, el in enumerate(elements):
        for i, sensor in enumerate(sensors):
            scale = 1.0 + 0.05 * el_idx
            A_sim = 5e-4 * scale
            B_sim = 1e-4 * scale

            decay = A_sim * np.exp(-time_array / tau_sim) + B_sim

            I_vals = decay + rng.normal(0, noise_level, size=len(time_array))
            Q_vals = decay * 0.3 + rng.normal(0, noise_level, size=len(time_array))

            suffix = f"{el.name}_{i + 1}"
            data_vars[f"I_{suffix}"] = xr.DataArray(I_vals, dims=["time_array"])
            data_vars[f"Q_{suffix}"] = xr.DataArray(Q_vals, dims=["time_array"])

    return xr.Dataset(data_vars, coords={"time_array": time_array})
