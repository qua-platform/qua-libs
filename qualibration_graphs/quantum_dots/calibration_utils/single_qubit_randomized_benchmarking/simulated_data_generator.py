"""Synthetic single-qubit randomized benchmarking datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    avg_physical_gates_per_clifford,
    decomposition_type,
)
from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_NATIVE_GATE_FIDELITY = 0.9985
_DEFAULT_AMPLITUDE = 0.44
_DEFAULT_OFFSET = 0.50


def _resolve_native_gate_fidelity(qubit, index: int) -> float:
    fidelity_dict = getattr(qubit, "gate_fidelity", None)
    stored = np.nan
    if isinstance(fidelity_dict, dict):
        stored = float(fidelity_dict.get("averaged", np.nan))
    fallback = _DEFAULT_NATIVE_GATE_FIDELITY - 2.0e-4 * (index % 4)
    fidelity = max(stored, fallback) if np.isfinite(stored) else fallback
    return float(np.clip(fidelity, 0.992, 0.9997))


def _alpha_from_native_gate_fidelity(native_gate_fidelity: float, avg_gates_per_clifford_value: float) -> float:
    epg = max(0.0, 1.0 - native_gate_fidelity)
    alpha_gate = np.clip(1.0 - 2.0 * epg, 1e-6, 1.0)
    return float(alpha_gate**avg_gates_per_clifford_value)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic RB survival probabilities matching the real dataset schema."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    depths = node.parameters.get_depths().astype(int)
    num_circuits = int(node.parameters.num_circuits_per_length)
    num_shots = int(node.parameters.num_shots)
    qubit_names = qubits.get_names()
    avg_gates = avg_physical_gates_per_clifford(decomposition_type)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "circuit": xr.DataArray(np.arange(num_circuits), attrs={"long_name": "circuit index"}),
        "depth": xr.DataArray(depths, attrs={"long_name": "number of Cliffords"}),
    }

    state = np.empty((len(qubits), num_circuits, len(depths)), dtype=float)
    for index, qubit in enumerate(qubits):
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        native_gate_fidelity = _resolve_native_gate_fidelity(qubit, index)
        alpha = _alpha_from_native_gate_fidelity(native_gate_fidelity, avg_gates)
        amplitude = _DEFAULT_AMPLITUDE - 0.02 * (index % 2)
        offset = _DEFAULT_OFFSET + 0.015 * (index % 3)

        base_survival = offset + amplitude * alpha**depths

        for circuit_index in range(num_circuits):
            circuit_alpha = float(np.clip(alpha + qubit_rng.normal(0.0, 6e-4), 1e-6, 0.99999))
            circuit_amp = amplitude * (1.0 + qubit_rng.normal(0.0, 0.04))
            circuit_offset = offset + qubit_rng.normal(0.0, 0.008)
            probability = circuit_offset + circuit_amp * circuit_alpha**depths
            probability += 0.004 * np.sin(2.0 * np.pi * depths / max(depths[-1], 2) + 0.15 * circuit_index)
            probability = np.clip(probability, 0.02, 0.98)
            counts = qubit_rng.binomial(num_shots, probability)
            state[index, circuit_index] = counts / max(num_shots, 1)

        # Keep the deepest depths monotonic on average even with circuit scatter.
        state[index] = np.clip(0.75 * state[index] + 0.25 * base_survival[None, :], 0.0, 1.0)

    return xr.Dataset(
        {"state": (["qubit", "circuit", "depth"], state)},
        coords={
            "qubit": qubit_names,
            "circuit": np.arange(num_circuits),
            "depth": depths,
            "n": np.array([0], dtype=int),
        },
    )
