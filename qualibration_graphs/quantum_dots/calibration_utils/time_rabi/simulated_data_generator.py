"""Synthetic 1D time-Rabi datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _damped_rabi(
    x: np.ndarray,
    *,
    center: float,
    frequency: float,
    decay: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    envelope = np.exp(-decay * np.abs(x - center))
    return 0.5 + 0.45 * envelope * np.sin(2.0 * np.pi * frequency * (x - center))


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated time-Rabi data so the full analysis pipeline can run without hardware."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    durations_ns = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
        dtype=float,
    )
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "pulse_duration": xr.DataArray(durations_ns, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    rng = np.random.default_rng(42)
    state = np.empty((len(qubits), len(durations_ns)), dtype=float)

    for index, _qubit in enumerate(qubits):
        # Use a short-enough pi-time so the Rabi frequency clears the FFT peak-fit lower clip in analysis.
        t_pi = 180.0 + 20.0 * (index % 4)
        probability = _damped_rabi(
            durations_ns,
            center=0.0,
            frequency=1.0 / (2.0 * t_pi),
            decay=1.0 / max(4000.0 + 80.0 * index, 400.0),
        )
        state[index] = np.clip(
            probability + rng.normal(0.0, 0.015, probability.shape),
            0.0,
            1.0,
        )

    return xr.Dataset(
        {"state": (["qubit", "pulse_duration"], state)},
        coords={
            "qubit": qubit_names,
            "pulse_duration": durations_ns,
            "n": np.array([0], dtype=int),
        },
    )
