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
_DEFAULT_T2_XY8_NS = 32_000.0  # 8 µs
_IDLE_FACTOR = 16.0


def _default_t2_xy8(_qubit, _index: int) -> float:
    return _DEFAULT_T2_XY8_NS


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
    """Generate synthetic XY8 raw streams.

    Produces averaged post-readout streams ``p_{qubit}`` on the τ axis.
    The underlying signal is a single exponential decay
    ``P(τ) = offset + A·exp(−16τ / T₂_XY8)``, matching :func:`fit_raw_data`.

    When ``parity_measurement`` is enabled, joint-outcome count streams are
    synthesised so that ``E_p1_given_p0_0`` equals the same decay model.
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

    rng = np.random.default_rng(seed=42)
    noise_std = float(getattr(node.parameters, "sim_noise_std", 0.03))
    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}

    for index, qubit in enumerate(qubits):
        t2 = _default_t2_xy8(qubit, index)
        amp = _DEFAULT_AMPLITUDE * (1.0 - 0.05 * index)
        off = _DEFAULT_OFFSET
        signal = _xy8_decay(tau_values, t2, amplitude=amp, offset=off)
        signal = signal + rng.normal(0.0, noise_std, size=signal.shape)
        signal = np.clip(signal, 0.0, 1.0)

        if node.parameters.parity_measurement:
            empty_weight = 0.7
            data_vars[f"p0_p0_{qubit.name}"] = (["tau"], np.full_like(signal, empty_weight))
            data_vars[f"p0_p1_{qubit.name}"] = (["tau"], empty_weight * signal)
            data_vars[f"p1_p0_{qubit.name}"] = (["tau"], np.full_like(signal, 0.1))
            data_vars[f"p1_p1_{qubit.name}"] = (["tau"], 0.1 * signal)
        else:
            data_vars[f"p_{qubit.name}"] = (["tau"], signal)

    tau_coord = xr.DataArray(tau_values, dims="tau", attrs=tau_attrs)
    data_arrays = {
        name: xr.DataArray(values, dims=dims, coords={"tau": tau_coord}) for name, (dims, values) in data_vars.items()
    }
    return xr.Dataset(data_arrays)
