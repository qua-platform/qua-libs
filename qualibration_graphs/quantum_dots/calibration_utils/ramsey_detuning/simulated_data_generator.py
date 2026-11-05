"""Synthetic Ramsey detuning-sweep datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_T2_STAR_NS = 2_800.0
_DEFAULT_BASELINE = 0.5


def _resolve_t2_star_ns(qubit, tau_ns: np.ndarray, index: int) -> float:
    stored_t2_s = float(getattr(qubit, "T2ramsey", np.nan))
    stored_t2_ns = stored_t2_s * 1e9 if np.isfinite(stored_t2_s) and stored_t2_s > 0 else np.nan
    tau_span = float(np.max(tau_ns) - np.min(tau_ns)) if len(tau_ns) > 1 else _DEFAULT_T2_STAR_NS
    fallback_t2_ns = max(8.0 * np.max(tau_ns), 1_500.0) + 250.0 * index
    t2_ns = stored_t2_ns if np.isfinite(stored_t2_ns) else fallback_t2_ns
    return float(np.clip(t2_ns, 150.0, 40.0 * max(np.max(tau_ns), 1.0)))


def _ramsey_detuning_trace(
    detuning_hz: np.ndarray,
    *,
    resonance_hz: float,
    tau_ns: float,
    t2_star_ns: float,
    amplitude: float,
    baseline: float,
) -> np.ndarray:
    tau_eff_ns = tau_ns + 32.0
    phase = 2.0 * np.pi * (detuning_hz - resonance_hz) * tau_eff_ns * 1e-9
    envelope = np.exp(-tau_ns / t2_star_ns)
    return baseline + amplitude * envelope * np.cos(phase)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic two-tau Ramsey detuning data matching the real dataset schema."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    tau_values = np.array(
        [
            node.parameters.idle_time_ns,
            node.parameters.idle_time_long_ns,
        ],
        dtype=float,
    )
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * 1e6,
        node.parameters.detuning_span_in_mhz / 2 * 1e6,
        node.parameters.detuning_step_in_mhz * 1e6,
        dtype=float,
    )
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "tau": xr.DataArray(tau_values, attrs={"long_name": "idle time", "units": "ns"}),
        "detuning": xr.DataArray(detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}),
    }

    state = np.empty((len(qubits), len(detuning_values), len(tau_values)), dtype=float)
    for index, qubit in enumerate(qubits):
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        t2_star_ns = _resolve_t2_star_ns(qubit, tau_values, index)
        resonance_hz = (0.14 + 0.03 * (index % 4)) * node.parameters.detuning_span_in_mhz * 1e6
        amplitude = 0.34 + 0.03 * (index % 2)
        baseline = _DEFAULT_BASELINE + 0.015 * (index % 3)

        for tau_index, tau_ns in enumerate(tau_values):
            trace = _ramsey_detuning_trace(
                detuning_values,
                resonance_hz=resonance_hz,
                tau_ns=float(tau_ns),
                t2_star_ns=t2_star_ns,
                amplitude=amplitude,
                baseline=baseline,
            )
            trace += 0.01 * np.sin(2.0 * np.pi * detuning_values / (1.6e6 + 0.2e6 * index))
            trace += qubit_rng.normal(0.0, 0.012, size=trace.shape)
            state[index, :, tau_index] = np.clip(trace, 0.0, 1.0)

    return xr.Dataset(
        {"state": (["qubit", "detuning", "tau"], state)},
        coords={
            "qubit": qubit_names,
            "tau": tau_values,
            "detuning": detuning_values,
            "n": np.array([0], dtype=int),
        },
    )
