"""Synthetic Ramsey datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_T2_STAR_NS = 8_000.0
_DEFAULT_AMPLITUDE = 0.34
_DEFAULT_OFFSET = 0.48


def _resolve_t2_star_ns(qubit, tau_ns: np.ndarray, index: int) -> float:
    stored_t2_s = float(getattr(qubit, "T2ramsey", np.nan))
    stored_t2_ns = stored_t2_s * 1e9 if np.isfinite(stored_t2_s) and stored_t2_s > 0 else np.nan
    tau_span = float(tau_ns[-1] - tau_ns[0]) if len(tau_ns) > 1 else _DEFAULT_T2_STAR_NS
    tau_step = float(np.min(np.diff(tau_ns))) if len(tau_ns) > 1 else max(tau_span / 20.0, 16.0)
    fallback_t2_ns = max(_DEFAULT_T2_STAR_NS, 0.3 * tau_span) + 1_000.0 * index
    t2_ns = max(stored_t2_ns, fallback_t2_ns) if np.isfinite(stored_t2_ns) else fallback_t2_ns
    return float(np.clip(t2_ns, max(8.0 * tau_step, 500.0), 8.0 * max(tau_span, 1.0)))


def _ramsey_trace(
    tau_ns: np.ndarray,
    *,
    ramsey_freq_hz: float,
    t2_star_ns: float,
    amplitude: float,
    offset: float,
    phase_rad: float,
) -> np.ndarray:
    tau_s = np.asarray(tau_ns, dtype=float) * 1e-9
    envelope = np.exp(-np.asarray(tau_ns, dtype=float) / t2_star_ns)
    phase = 2.0 * np.pi * ramsey_freq_hz * tau_s + phase_rad
    return offset + amplitude * envelope * np.cos(phase)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic ±delta Ramsey data matching the real dataset schema."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    detuning_hz = float(node.parameters.frequency_detuning_in_mhz) * 1e6
    detuning_values = np.array([detuning_hz, -detuning_hz], dtype=float)
    tau_values = get_idle_times_in_clock_cycles(node.parameters).astype(float) * 4.0
    qubit_names = qubits.get_names()

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubit_names),
        "detuning": xr.DataArray(detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}),
        "tau": xr.DataArray(tau_values, attrs={"long_name": "idle time", "units": "ns"}),
    }

    state = np.empty((len(qubits), len(detuning_values), len(tau_values)), dtype=float)
    for index, qubit in enumerate(qubits):
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        t2_star_ns = _resolve_t2_star_ns(qubit, tau_values, index)
        freq_offset_hz = (0.08 + 0.03 * (index % 3)) * detuning_hz
        amplitude = _DEFAULT_AMPLITUDE - 0.03 * (index % 2)
        offset = _DEFAULT_OFFSET + 0.015 * (index % 3)
        phase = 0.02 * (index % 3)

        for detuning_index, applied_detuning_hz in enumerate(detuning_values):
            ramsey_freq_hz = abs(freq_offset_hz - applied_detuning_hz)
            signal = _ramsey_trace(
                tau_values,
                ramsey_freq_hz=ramsey_freq_hz,
                t2_star_ns=t2_star_ns,
                amplitude=amplitude,
                offset=offset,
                phase_rad=phase if detuning_index == 0 else -phase,
            )
            signal += 0.003 * np.sin(2.0 * np.pi * tau_values / (9_000.0 + 500.0 * index))
            signal += qubit_rng.normal(0.0, 0.01, size=signal.shape)
            state[index, detuning_index] = np.clip(signal, 0.0, 1.0)

    return xr.Dataset(
        {"state": (["qubit", "detuning", "tau"], state)},
        coords={
            "qubit": qubit_names,
            "detuning": detuning_values,
            "tau": tau_values,
            "n": np.array([0], dtype=int),
        },
    )
