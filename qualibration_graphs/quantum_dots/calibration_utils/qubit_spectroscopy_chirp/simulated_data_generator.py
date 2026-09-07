from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from qualang_tools.units import unit
from qualibration_libs.parameters.experiment import get_qubits

u = unit(coerce_to_integer=True)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated chirped spectroscopy ``state`` data.

    For each qubit the Larmor frequency from the QUAM state determines where
    the thresholded state response appears. Because a chirp drive is broad and
    rough, the peak is deliberately coarse: a random 1-4 bin plateau at high
    signal, with everything else near zero.

    Parameters
    ----------
    node : QualibrationNode
        Node with ``parameters`` and ``machine`` already set.
        Writes ``qubits`` into ``node.namespace``.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)
    detuning_centers = dfs + step / 2

    rng = np.random.default_rng(seed=42)
    state_rows = []

    for q in qubits:
        true_detuning = q.larmor_frequency - q.xy.RF_frequency

        # Build a coarse thresholded-state response versus chirp-band centre.
        signal = rng.uniform(0.02, 0.08, size=len(dfs))
        nearest_idx = int(np.argmin(np.abs(detuning_centers - true_detuning)))
        n_peak = rng.integers(1, 3)  # 1 to 4 inclusive
        half = n_peak // 2
        start = max(0, nearest_idx - half)
        end = min(len(dfs), start + n_peak)
        signal[start:end] = rng.uniform(0.4, 0.8, size=end - start)
        state_rows.append(signal)

    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.asarray(state_rows, dtype=float),
                dims=["qubit", "detuning"],
                coords={
                    "qubit": [q.name for q in qubits],
                    "detuning": detuning_centers,
                },
            )
        }
    )
