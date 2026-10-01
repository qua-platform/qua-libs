"""Generate synthetic XY8 dynamical decoupling datasets for offline analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

_DEFAULT_AMPLITUDE = 0.65
_DEFAULT_OFFSET = 0.05
DEFAULT_T2_XY8_NS = 32_000.0  # 32 µs
_IDLE_FACTOR = 16.0


def _xy8_decay(
    tau_ns: np.ndarray,
    t2_xy8_ns: float,
    *,
    amplitude: float = _DEFAULT_AMPLITUDE,
    offset: float = _DEFAULT_OFFSET,
) -> np.ndarray:
    """P(τ) = offset + A·exp(−16τ / T₂_XY8)."""
    return offset + amplitude * np.exp(-_IDLE_FACTOR * tau_ns / t2_xy8_ns)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate synthetic XY8 state data.

    Produces a ``state(qubit, tau)`` array whose underlying signal is a single
    exponential decay ``P(τ) = offset + A·exp(−16τ / T₂_XY8)``, matching
    :func:`fit_raw_data`.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )
    tau_attrs = {
        "long_name": "XY8 CPMG half-spacing τ (bookend τ, inter-pulse 2τ)",
        "units": "ns",
    }

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(tau_values, attrs=tau_attrs),
    }

    noise_std: float = 0.03
    state_rows = []

    for qubit in qubits:
        qubit_rng = np.random.default_rng(seed=42 + sum(map(ord, qubit.name)))
        t2 = DEFAULT_T2_XY8_NS
        amp = _DEFAULT_AMPLITUDE
        off = _DEFAULT_OFFSET
        signal = _xy8_decay(tau_values, t2, amplitude=amp, offset=off)
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
