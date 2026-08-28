from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualang_tools.units import unit
from qualibration_libs.parameters import get_qubits

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
    """Generate 08b-style raw spectroscopy data for the real analysis pipeline.

    The returned dataset matches the raw handle layout produced by the real OPX
    execution path:

    - explicitly named parity streams, e.g. ``p_q1_parity_diff`` or
      ``p0_p1_q1_parity_diff``
    - explicitly named averaged raw IQ traces, e.g. ``I_q1_raw`` and
      ``Q_q1_raw``

    This allows ``process_raw_dataset()``, ``fit_raw_data()``, and
    ``plot_all()`` to run unchanged on the simulated output.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step, dtype=float)
    if len(dfs) == 0:
        dfs = np.array([0.0])

    rng = np.random.default_rng(seed=42)
    data_vars: dict[str, xr.DataArray] = {}
    coords = {
        "detuning": xr.DataArray(
            dfs,
            dims="detuning",
            attrs={"long_name": "drive frequency", "units": "Hz"},
        )
    }

    default_fwhm = 3.0 * u.MHz
    width = max(float(default_fwhm), 6.0 * float(step), 1.0)

    for index, qubit in enumerate(qubits):
        qname = qubit.name
        true_detuning = float(qubit.larmor_frequency - qubit.xy.RF_frequency)

        resonance = _lorentzian(dfs, true_detuning, width)
        quadrature = _dispersive(dfs, true_detuning, width)

        analysis_signal = np.clip(
            0.05 + 0.55 * resonance + rng.normal(scale=0.01, size=len(dfs)),
            0.0,
            1.0,
        )
        secondary_branch = np.clip(
            0.45 + 0.04 * quadrature + rng.normal(scale=0.01, size=len(dfs)),
            0.0,
            1.0,
        )

        if node.parameters.parity_measurement:
            data_vars[f"p0_p0_{qname}_parity_diff"] = xr.DataArray(
                1.0 - analysis_signal,
                dims="detuning",
                coords=coords,
            )
            data_vars[f"p0_p1_{qname}_parity_diff"] = xr.DataArray(
                analysis_signal,
                dims="detuning",
                coords=coords,
            )
            data_vars[f"p1_p0_{qname}_parity_diff"] = xr.DataArray(
                1.0 - secondary_branch,
                dims="detuning",
                coords=coords,
            )
            data_vars[f"p1_p1_{qname}_parity_diff"] = xr.DataArray(
                secondary_branch,
                dims="detuning",
                coords=coords,
            )
        else:
            data_vars[f"p_{qname}_parity_diff"] = xr.DataArray(
                analysis_signal,
                dims="detuning",
                coords=coords,
            )

        i_trace = 0.02 * index + 0.18 * resonance + rng.normal(scale=0.004, size=len(dfs))
        q_trace = -0.015 * index + 0.10 * quadrature + rng.normal(scale=0.004, size=len(dfs))

        data_vars[f"I_{qname}_raw"] = xr.DataArray(
            i_trace,
            dims="detuning",
            coords=coords,
        )
        data_vars[f"Q_{qname}_raw"] = xr.DataArray(
            q_trace,
            dims="detuning",
            coords=coords,
        )

    return xr.Dataset(data_vars)
