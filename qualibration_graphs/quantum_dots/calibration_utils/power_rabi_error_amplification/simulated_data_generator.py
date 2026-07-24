"""Synthetic error-amplified power-Rabi datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _assign_stream(
    data_vars: dict,
    qname: str,
    probability: np.ndarray,
    dims: tuple[str, ...],
    *,
    parity_measurement: bool,
    rng: np.random.Generator,
    noise_scale: float = 0.02,
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
    """Generate simulated error-amplified power-Rabi data for offline validation."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
        dtype=float,
    )
    n_pulses = np.arange(2, node.parameters.max_n_pulses, 2, dtype=float)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_pulses": xr.DataArray(n_pulses, attrs={"long_name": "number of pi pulses"}),
        "amp_prefactor": xr.DataArray(
            amps, attrs={"long_name": "pulse amplitude prefactor"}
        ),
    }

    rng = np.random.default_rng(42)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    for index, qubit in enumerate(qubits):
        a_pi = 1.0 + 0.02 * (index % 5)
        scale = 2.8 + 0.15 * (index % 3)
        envelope = np.exp(-0.004 * n_pulses)[:, None]
        phase = scale * n_pulses[:, None] * (amps[None, :] - a_pi)
        probability = np.sin(phase) ** 2 * envelope
        probability[0, :] = 0.0
        _assign_stream(
            data_vars,
            qubit.name,
            probability,
            ("n_pulses", "amp_prefactor"),
            parity_measurement=node.parameters.parity_measurement,
            rng=rng,
        )

    return xr.Dataset(
        {name: (dims, values) for name, (dims, values) in data_vars.items()},
        coords={
            "n_pulses": n_pulses,
            "amp_prefactor": amps,
            "n": np.array([0], dtype=int),
        },
    )
