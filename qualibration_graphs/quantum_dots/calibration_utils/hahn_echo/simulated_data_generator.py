"""Generate synthetic Hahn echo datasets for offline analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

# Typical spin-echo contrast (matches analysis tests).
_DEFAULT_AMPLITUDE = 0.65
_DEFAULT_OFFSET = 0.05
DEFAULT_T2_ECHO_NS = 2_000.0  # 2 µs
_IDLE_FACTOR = 2.0


def _echo_decay(
    tau_ns: np.ndarray,
    t2_ns: float,
    *,
    amplitude: float = _DEFAULT_AMPLITUDE,
    offset: float = _DEFAULT_OFFSET,
) -> np.ndarray:
    """P(τ) = offset + A·exp(−2τ / T₂_echo)."""
    return offset + amplitude * np.exp(-_IDLE_FACTOR * tau_ns / t2_ns)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic Hahn echo state data.

    Produces a ``state(qubit, tau)`` array whose underlying signal is a single
    exponential decay ``P(τ) = offset + A·exp(−2τ / T₂_echo)``, matching
    :func:`fit_raw_data`.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )
    tau_attrs = {
        "long_name": "Hahn echo idle delay τ (each x90–y180 segment)",
        "units": "ns",
    }

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(tau_values, attrs=tau_attrs),
    }

    noise_std = float(getattr(node.parameters, "sim_noise_std", 0.03))
    state_rows = []

    for qubit in qubits:
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        t2 = DEFAULT_T2_ECHO_NS
        amp = _DEFAULT_AMPLITUDE
        off = _DEFAULT_OFFSET
        signal = _echo_decay(tau_values, t2, amplitude=amp, offset=off)
        signal = signal + qubit_rng.normal(0.0, noise_std, size=signal.shape)
        signal = np.clip(signal, 0.0, 1.0)
        state_rows.append(signal)

    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.asarray(state_rows, dtype=float),
                dims=("qubit", "tau"),
                coords={"qubit": qubits.get_names(), "tau": tau_values},
                attrs={"long_name": "thresholded qubit state"},
            )
        }
    )
