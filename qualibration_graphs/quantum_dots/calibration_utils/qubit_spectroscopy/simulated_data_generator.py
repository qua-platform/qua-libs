from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualang_tools.units import unit
from qualibration_libs.parameters.experiment import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

u = unit(coerce_to_integer=True)

__all__ = ["generate_simulated_dataset"]


def _lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Return a unit-height Lorentzian profile."""
    hwhm = max(float(fwhm) / 2.0, 1.0)
    return 1.0 / (1.0 + ((x - center) / hwhm) ** 2)


def _dispersive(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Return a simple dispersive companion trace for the Q quadrature."""
    hwhm = max(float(fwhm) / 2.0, 1.0)
    scaled = (x - center) / hwhm
    return scaled / (1.0 + scaled**2)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate 08b-style spectroscopy data for the real analysis pipeline.

    The returned dataset matches the post-fetch layout produced by the real OPX
    execution path: thresholded ``state(qubit, detuning)`` plus raw ``I`` and
    ``Q`` traces with the same dimensions. This allows ``process_raw_dataset()``,
    ``fit_raw_data()``, and ``plot_all()`` to run unchanged on the simulated output.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step, dtype=float)
    if len(dfs) == 0:
        dfs = np.array([0.0])

    rng = np.random.default_rng(seed=42)
    state_rows = []
    i_rows = []
    q_rows = []

    default_fwhm = 3.0 * u.MHz
    width = max(float(default_fwhm), 6.0 * float(step), 1.0)

    for index, qubit in enumerate(qubits):
        true_detuning = float(qubit.larmor_frequency - qubit.xy.RF_frequency)

        resonance = _lorentzian(dfs, true_detuning, width)
        quadrature = _dispersive(dfs, true_detuning, width)

        state_trace = np.clip(
            0.05 + 0.55 * resonance + rng.normal(scale=0.01, size=len(dfs)),
            0.0,
            1.0,
        )
        state_rows.append(state_trace)

        i_trace = 0.02 * index + 0.18 * resonance + rng.normal(scale=0.004, size=len(dfs))
        q_trace = -0.015 * index + 0.10 * quadrature + rng.normal(scale=0.004, size=len(dfs))
        i_rows.append(i_trace)
        q_rows.append(q_trace)

    coords = {
        "qubit": [q.name for q in qubits],
        "detuning": xr.DataArray(dfs, dims="detuning", attrs={"long_name": "drive frequency", "units": "Hz"}),
    }
    return xr.Dataset(
        {
            "state": xr.DataArray(np.asarray(state_rows, dtype=float), dims=["qubit", "detuning"], coords=coords),
            "I": xr.DataArray(np.asarray(i_rows, dtype=float), dims=["qubit", "detuning"], coords=coords),
            "Q": xr.DataArray(np.asarray(q_rows, dtype=float), dims=["qubit", "detuning"], coords=coords),
        }
    )
