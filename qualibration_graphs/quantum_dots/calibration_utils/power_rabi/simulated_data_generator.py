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


def _assign_stream(
    data_vars: dict,
    qname: str,
    probability: np.ndarray,
    dims: tuple[str, ...],
    *,
    parity_measurement: bool,
    rng: np.random.Generator,
    noise_scale: float = 0.015,
) -> None:
    prob = np.clip(
        np.asarray(probability, dtype=float) + rng.normal(0.0, noise_scale, probability.shape),
        0.0,
        1.0,
    )
    if parity_measurement:
        leak = 0.01
        data_vars[f"p1_p0_{qname}"] = (dims, prob)
        data_vars[f"p0_p0_{qname}"] = (dims, np.clip(1.0 - prob, 0.0, 1.0))
        data_vars[f"p0_p1_{qname}"] = (dims, leak * (1.0 - prob))
        data_vars[f"p1_p1_{qname}"] = (dims, leak * prob)
    else:
        data_vars[f"p_{qname}"] = (dims, prob)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated power-Rabi data so the full analysis pipeline can run without hardware."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
        dtype=float,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    rng = np.random.default_rng(42)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    for index, qubit in enumerate(qubits):
        probability = _power_rabi_probability(
            amps,
            rabi_frequency=3.2 + 0.2 * (index % 3),
            amplitude=0.98 - 0.02 * (index % 2),
            decay=0.04 + 0.01 * (index % 2),
        )
        _assign_stream(
            data_vars,
            qubit.name,
            probability,
            ("amp_prefactor",),
            parity_measurement=node.parameters.parity_measurement,
            rng=rng,
        )

    return xr.Dataset(
        {name: (dims, values) for name, (dims, values) in data_vars.items()},
        coords={"amp_prefactor": amps, "n": np.array([0], dtype=int)},
    )
