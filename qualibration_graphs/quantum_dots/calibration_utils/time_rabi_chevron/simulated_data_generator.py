"""Synthetic Rabi-chevron datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from qualang_tools.units import unit

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _rabi_chevron(
    detuning_hz: np.ndarray,
    duration_ns: np.ndarray,
    *,
    omega_hz: float,
    contrast: float,
    baseline: float,
) -> np.ndarray:
    det = detuning_hz[:, None].astype(float)
    dur = duration_ns[None, :].astype(float)
    delta = 2.0 * np.pi * det
    omega = 2.0 * np.pi * omega_hz
    t_s = dur * 1e-9
    omega_eff = np.sqrt(omega**2 + delta**2)
    envelope = omega**2 / (omega**2 + delta**2)
    oscillation = np.sin(0.5 * omega_eff * t_s) ** 2
    return baseline + contrast * envelope * oscillation


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated Rabi-chevron data so the full analysis pipeline can run without hardware."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)

    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
        dtype=float,
    )
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    detuning = np.arange(-span // 2, span // 2, step, dtype=float)
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "detuning": xr.DataArray(detuning, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    rng = np.random.default_rng(42)
    state = np.empty((len(qubits), len(detuning), len(pulse_durations)), dtype=float)

    for index, _qubit in enumerate(qubits):
        probability = _rabi_chevron(
            detuning,
            pulse_durations,
            omega_hz=2e6 + 0.5e6 * (index % 3),
            contrast=0.35 + 0.03 * (index % 2),
            baseline=0.08 + 0.01 * (index % 2),
        )
        state[index] = np.clip(
            probability + rng.normal(0.0, 0.012, probability.shape),
            0.0,
            1.0,
        )

    return xr.Dataset(
        {"state": (["qubit", "detuning", "pulse_duration"], state)},
        coords={
            "qubit": qubit_names,
            "detuning": detuning,
            "pulse_duration": pulse_durations,
            "n": np.array([0], dtype=int),
        },
    )
