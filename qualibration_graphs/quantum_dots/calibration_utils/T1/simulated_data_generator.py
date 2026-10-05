"""Synthetic T1 datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_T1_NS = 6_000.0
_DEFAULT_AMPLITUDE = 0.78
_DEFAULT_OFFSET = 0.08


def _t1_decay(
    tau_ns: np.ndarray,
    *,
    t1_ns: float,
    amplitude: float,
    offset: float,
) -> np.ndarray:
    tau_ns = np.asarray(tau_ns, dtype=float)
    return offset + amplitude * np.exp(-tau_ns / t1_ns)


def _resolve_t1_ns(qubit, tau_values: np.ndarray, index: int) -> float:
    stored_t1_s = float(getattr(qubit, "T1", np.nan))
    stored_t1_ns = stored_t1_s * 1e9 if np.isfinite(stored_t1_s) and stored_t1_s > 0 else np.nan
    tau_span = float(tau_values[-1] - tau_values[0]) if len(tau_values) > 1 else _DEFAULT_T1_NS
    tau_step = float(np.min(np.diff(tau_values))) if len(tau_values) > 1 else max(tau_span / 20.0, 16.0)
    fallback_t1_ns = max(0.8 * tau_span, 2_000.0) + 700.0 * index
    t1_ns = stored_t1_ns if np.isfinite(stored_t1_ns) else fallback_t1_ns
    return float(np.clip(t1_ns, max(8.0 * tau_step, 250.0), 8.0 * max(tau_span, 1.0)))


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic T1 state data matching the real dataset schema."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
        dtype=float,
    )
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "tau": xr.DataArray(tau_values, attrs={"long_name": "idle time", "units": "ns"}),
    }

    state = np.empty((len(qubits), len(tau_values)), dtype=float)
    for index, qubit in enumerate(qubits):
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        t1_ns = _resolve_t1_ns(qubit, tau_values, index)
        amplitude = _DEFAULT_AMPLITUDE - 0.04 * (index % 3)
        offset = _DEFAULT_OFFSET + 0.01 * (index % 2)
        signal = _t1_decay(
            tau_values,
            t1_ns=t1_ns,
            amplitude=amplitude,
            offset=offset,
        )
        signal += 0.01 * np.exp(-tau_values / max(t1_ns * 0.35, 200.0)) * np.sin(2.0 * np.pi * tau_values / 900.0)
        signal += qubit_rng.normal(0.0, 0.012, size=signal.shape)
        state[index] = np.clip(signal, 0.0, 1.0)

    return xr.Dataset(
        {"state": (["qubit", "tau"], state)},
        coords={
            "qubit": qubit_names,
            "tau": tau_values,
            "n": np.array([0], dtype=int),
        },
    )
