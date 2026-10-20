"""Synthetic Ramsey chevron datasets for offline analysis validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_T2_STAR_NS = 50_000.0
_PHASE_SCALE = 0.22


def _resolve_t2_star_ns(qubit, tau_ns: np.ndarray, index: int) -> float:
    stored_t2_s = float(getattr(qubit, "T2ramsey", np.nan))
    stored_t2_ns = stored_t2_s * 1e9 if np.isfinite(stored_t2_s) and stored_t2_s > 0 else np.nan
    tau_span = float(tau_ns[-1] - tau_ns[0]) if len(tau_ns) > 1 else _DEFAULT_T2_STAR_NS
    tau_step = float(np.min(np.diff(tau_ns))) if len(tau_ns) > 1 else max(tau_span / 20.0, 16.0)
    # For simulated chevrons we prefer a visibly coherent pattern over the often-short
    # machine-default T2ramsey stored in the QuAM. Use the stored value only as a floor.
    fallback_t2_ns = max(_DEFAULT_T2_STAR_NS, 1.2 * tau_span) + 5_000.0 * index
    t2_ns = max(stored_t2_ns, fallback_t2_ns) if np.isfinite(stored_t2_ns) else fallback_t2_ns
    return float(np.clip(t2_ns, max(8.0 * tau_step, 150.0), 10.0 * max(tau_span, 1.0)))


def _effective_t2_star(gamma: float, sigma_g: float) -> float:
    if sigma_g < 1e-12:
        return 1.0 / gamma if gamma > 1e-12 else np.nan
    discriminant = gamma**2 + 4.0 * sigma_g**2
    return (-gamma + np.sqrt(discriminant)) / (2.0 * sigma_g**2)


def _solve_gamma_sigma(target_t2_ns: float) -> tuple[float, float]:
    gamma = 0.18 / target_t2_ns
    sigma_g = 0.22 / target_t2_ns
    scale = _effective_t2_star(gamma, sigma_g) / target_t2_ns
    if np.isfinite(scale) and scale > 0:
        gamma /= scale
        sigma_g /= scale
    return gamma, sigma_g


def _ramsey_chevron(
    detuning_hz: np.ndarray,
    tau_ns: np.ndarray,
    *,
    resonance_hz: float,
    amplitude: float,
    baseline: float,
    gamma: float,
    sigma_g: float,
    detuning_width_hz: float,
) -> np.ndarray:
    detuning = np.asarray(detuning_hz, dtype=float)[:, None]
    tau = np.asarray(tau_ns, dtype=float)[None, :]
    phase = 2.0 * np.pi * (detuning - resonance_hz) * (_PHASE_SCALE * tau) * 1e-9
    envelope = np.exp(-gamma * tau - (sigma_g * tau) ** 2)
    detuning_envelope = np.exp(-((detuning - resonance_hz) / detuning_width_hz) ** 2)
    return baseline + amplitude * envelope * detuning_envelope * np.cos(phase)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic Ramsey chevron data matching the real dataset schema."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    tau_values = get_idle_times_in_clock_cycles(node.parameters).astype(float) * 4.0
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * 1e6,
        node.parameters.detuning_span_in_mhz / 2 * 1e6,
        node.parameters.detuning_step_in_mhz * 1e6,
        dtype=float,
    )
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
        gamma, sigma_g = _solve_gamma_sigma(t2_star_ns)
        resonance_hz = (-0.1 + 0.08 * (index % 4)) * node.parameters.detuning_span_in_mhz * 1e6
        detuning_width_hz = (0.32 + 0.03 * (index % 3)) * node.parameters.detuning_span_in_mhz * 1e6
        amplitude = 0.36 - 0.025 * (index % 2)
        baseline = 0.48 + 0.015 * (index % 3)
        chevron = _ramsey_chevron(
            detuning_values,
            tau_values,
            resonance_hz=resonance_hz,
            amplitude=amplitude,
            baseline=baseline,
            gamma=gamma,
            sigma_g=sigma_g,
            detuning_width_hz=detuning_width_hz,
        )
        detuning_drift = 0.004 * np.sin(2.0 * np.pi * detuning_values[:, None] / (2.4e6 + 0.2e6 * index))
        tau_drift = 0.003 * np.sin(2.0 * np.pi * tau_values[None, :] / (12_000.0 + 800.0 * index))
        chevron += detuning_drift + tau_drift
        chevron += qubit_rng.normal(0.0, 0.006, size=chevron.shape)
        state[index] = np.clip(chevron, 0.0, 1.0)

    return xr.Dataset(
        {"state": (["qubit", "detuning", "tau"], state)},
        coords={
            "qubit": qubit_names,
            "detuning": detuning_values,
            "tau": tau_values,
            "n": np.array([0], dtype=int),
        },
    )
