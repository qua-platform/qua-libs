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


def _assign_stream(
    data_vars: dict,
    qname: str,
    probability: np.ndarray,
    dims: tuple[str, ...],
    *,
    parity_measurement: bool,
    rng: np.random.Generator,
) -> None:
    prob = np.clip(
        np.asarray(probability, dtype=float) + rng.normal(0.0, 0.015, probability.shape),
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
    """Generate simulated time-Rabi data so the full analysis pipeline can run without hardware."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    durations_ns = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
        dtype=float,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "pulse_duration": xr.DataArray(durations_ns, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    rng = np.random.default_rng(42)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    for index, qubit in enumerate(qubits):
        # Use a short-enough pi-time so the Rabi frequency clears the FFT peak-fit lower clip in analysis.
        t_pi = 180.0 + 20.0 * (index % 4)
        probability = _damped_rabi(
            durations_ns,
            center=0.0,
            frequency=1.0 / (2.0 * t_pi),
            decay=1.0 / max(4000.0 + 80.0 * index, 400.0),
        )
        _assign_stream(
            data_vars,
            qubit.name,
            probability,
            ("pulse_duration",),
            parity_measurement=node.parameters.parity_measurement,
            rng=rng,
        )

    return xr.Dataset(
        {name: (dims, values) for name, (dims, values) in data_vars.items()},
        coords={"pulse_duration": durations_ns, "n": np.array([0], dtype=int)},
    )
