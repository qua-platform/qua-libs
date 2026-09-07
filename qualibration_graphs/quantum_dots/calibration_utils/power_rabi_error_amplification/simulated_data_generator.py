"""Synthetic error-amplified power-Rabi datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from calibration_utils.power_rabi_error_amplification.parameters import get_number_of_pulses
from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated error-amplified power-Rabi data for offline validation."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
        dtype=float,
    )
    n_pulses = np.asarray(get_number_of_pulses(node.parameters), dtype=float)
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "n_pulses": xr.DataArray(n_pulses, attrs={"long_name": "number of pi pulses"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    rng = np.random.default_rng(42)
    state = np.empty((len(qubits), len(n_pulses), len(amps)), dtype=float)

    for index, _qubit in enumerate(qubits):
        a_pi = 1.0 + 0.02 * (index % 5)
        scale = 2.8 + 0.15 * (index % 3)
        envelope = np.exp(-0.004 * n_pulses)[:, None]
        phase = scale * n_pulses[:, None] * (amps[None, :] - a_pi)
        probability = np.sin(phase) ** 2 * envelope
        probability[0, :] = 0.0
        state[index] = np.clip(
            probability + rng.normal(0.0, 0.02, probability.shape),
            0.0,
            1.0,
        )

    return xr.Dataset(
        {"state": (["qubit", "n_pulses", "amp_prefactor"], state)},
        coords={
            "qubit": qubit_names,
            "n_pulses": n_pulses,
            "amp_prefactor": amps,
            "n": np.array([0], dtype=int),
        },
    )
