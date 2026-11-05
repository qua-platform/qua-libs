"""Synthetic power-Rabi datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _power_rabi_probability(
    amp_prefactor: np.ndarray,
    *,
    rabi_frequency: float,
    amplitude: float,
    decay: float,
) -> np.ndarray:
    """Excited-state probability for a resonant power-Rabi amplitude sweep.

    At resonance the population follows Rabi's formula

        P(a) = sin²(Ω · a / 2),

    where the drive amplitude prefactor ``a`` is proportional to the Rabi
    frequency Ω (fixed pulse duration).  A weak exponential envelope mimics
    slow calibration drift.  ``rabi_frequency`` is Ω/(2π) in cycles per
    unit amplitude prefactor.
    """
    a = np.asarray(amp_prefactor, dtype=float)
    omega = 2.0 * np.pi * rabi_frequency
    envelope = np.exp(-decay * a)
    return np.clip(amplitude * envelope * np.sin(0.5 * omega * a) ** 2, 0.0, 1.0)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated power-Rabi data so the full analysis pipeline can run without hardware."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
        dtype=float,
    )
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    rng = np.random.default_rng(42)
    state = np.empty((len(qubits), len(amps)), dtype=float)

    for index, _qubit in enumerate(qubits):
        probability = _power_rabi_probability(
            amps,
            rabi_frequency=3.2 + 0.2 * (index % 3),
            amplitude=0.98 - 0.02 * (index % 2),
            decay=0.04 + 0.01 * (index % 2),
        )
        state[index] = np.clip(
            probability + rng.normal(0.0, 0.015, probability.shape),
            0.0,
            1.0,
        )

    return xr.Dataset(
        {"state": (["qubit", "amp_prefactor"], state)},
        coords={
            "qubit": qubit_names,
            "amp_prefactor": amps,
            "n": np.array([0], dtype=int),
        },
    )
