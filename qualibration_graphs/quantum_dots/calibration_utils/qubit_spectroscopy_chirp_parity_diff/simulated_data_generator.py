from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from qualang_tools.units import unit
from calibration_utils.common_utils.experiment import get_qubits

u = unit(coerce_to_integer=True)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated chirp spectroscopy parity-stream data.

    For each qubit the Larmor frequency from the QUAM state determines where
    the parity signal appears.  Because a chirp drive is broad and rough, the
    peak is deliberately coarse: a random 1-4 bin plateau at high signal, with
    everything else near zero.

    The returned dataset has the same variable layout as real OPX data after
    buffering — i.e. the four joint-outcome streams per qubit
    (``p0_p0_{qname}``, ``p0_p1_{qname}``, ``p1_p0_{qname}``,
    ``p1_p1_{qname}``) when ``parity_pre_measurement`` is ``True``, or a
    single ``p_{qname}`` when ``False``.  This ensures
    ``process_parity_streams`` and the rest of the analysis pipeline run
    unchanged.

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

    parity_pre_measurement = node.parameters.parity_pre_measurement
    rng = np.random.default_rng(seed=42)

    data_vars: dict[str, xr.DataArray] = {}
    det_coord = {"detuning": dfs}

    for q in qubits:
        true_detuning = q.larmor_frequency - q.xy.RF_frequency

        # Build the signal (E_p2_given_p1_0 equivalent)
        signal = rng.uniform(0.02, 0.08, size=len(dfs))
        nearest_idx = int(np.argmin(np.abs(dfs - true_detuning)))
        n_peak = rng.integers(1, 3)  # 1 to 4 inclusive
        half = n_peak // 2
        start = max(0, nearest_idx - half)
        end = min(len(dfs), start + n_peak)
        signal[start:end] = rng.uniform(0.4, 0.8, size=end - start)

        if parity_pre_measurement:
            # signal = E[p2=1 | p1=0] = p0_p1 / (p0_p0 + p0_p1)
            # Set p0_p0 + p0_p1 = 1  =>  p0_p1 = signal, p0_p0 = 1 - signal
            # p1 branch carries no information: 50/50
            data_vars[f"p0_p0_{q.name}"] = xr.DataArray(1.0 - signal, dims=["detuning"], coords=det_coord)
            data_vars[f"p0_p1_{q.name}"] = xr.DataArray(signal, dims=["detuning"], coords=det_coord)
            data_vars[f"p1_p0_{q.name}"] = xr.DataArray(np.full_like(signal, 0.5), dims=["detuning"], coords=det_coord)
            data_vars[f"p1_p1_{q.name}"] = xr.DataArray(np.full_like(signal, 0.5), dims=["detuning"], coords=det_coord)
        else:
            data_vars[f"p_{q.name}"] = xr.DataArray(signal, dims=["detuning"], coords=det_coord)

    return xr.Dataset(data_vars)
