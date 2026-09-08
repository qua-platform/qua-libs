from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from calibration_utils.iq_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)
from qualibration_libs.parameters.experiment import get_qubit_pairs

from .helper_utils import validate_and_build_arrays

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _detuning_unit(detuning: float, detuning_min: float, detuning_max: float) -> float:
    span = detuning_max - detuning_min
    if abs(span) < 1e-15:
        return 0.5
    return float(np.clip((detuning - detuning_min) / span, 0.0, 1.0))


def _buffer_unit(buffer_duration: float, buffer_min: float, buffer_max: float) -> float:
    span = buffer_max - buffer_min
    if abs(span) < 1e-15:
        return 0.5
    return float(np.clip((buffer_duration - buffer_min) / span, 0.0, 1.0))


def generate_simulated_dataset(node: "QualibrationNode") -> xr.Dataset:
    """Generate a synthetic 2D PSB sweep with contrast varying over detuning and buffer."""
    qubit_pairs = get_qubit_pairs(node)
    pair_names = [qp.name for qp in qubit_pairs]
    detuning_array, _buffer_cc_array, buffer_ns_array = validate_and_build_arrays(node)

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["detuning_array"] = detuning_array
    node.namespace["buffer_ns_array"] = buffer_ns_array
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(pair_names),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "detuning", "units": "V"}),
        "buffer_duration": xr.DataArray(
            buffer_ns_array.astype(float),
            attrs={"long_name": "buffer duration", "units": "ns"},
        ),
    }

    num_shots = int(node.parameters.num_shots)
    n_pairs = len(qubit_pairs)
    n_detuning = len(detuning_array)
    n_buffer = len(buffer_ns_array)

    i_arr = np.zeros((n_pairs, num_shots, n_detuning, n_buffer), dtype=float)
    q_arr = np.zeros((n_pairs, num_shots, n_detuning, n_buffer), dtype=float)

    tau_m = 1.0
    t1 = 2.0
    sigma_i_base = 0.12e-2
    sigma_q_base = 0.10e-2
    detuning_min = float(detuning_array.min())
    detuning_max = float(detuning_array.max())
    buffer_min = float(buffer_ns_array.min())
    buffer_max = float(buffer_ns_array.max())

    for pair_index, _qubit_pair in enumerate(qubit_pairs):
        rng = np.random.default_rng(seed=42 + pair_index * 9973)

        # Give each pair a stable readout axis in the IQ plane while keeping the
        # synthetic contrast envelope shared across the full 2D sweep.
        theta = 0.38 + 0.27 * float(pair_index)
        c_ax = float(np.cos(theta))
        s_ax = float(np.sin(theta))
        y_s_ref = (-1.25e-2) * (1.0 + 0.06 * float(pair_index))
        y_t_ref = (1.25e-2) * (1.0 + 0.06 * float(pair_index))

        for detuning_index, detuning in enumerate(detuning_array):
            detuning_weight = float(np.sin(np.pi * _detuning_unit(detuning, detuning_min, detuning_max)) ** 2)

            for buffer_index, buffer_duration in enumerate(buffer_ns_array.astype(float)):
                buffer_weight = 0.25 + 0.75 * _buffer_unit(buffer_duration, buffer_min, buffer_max)

                # The two-state separation opens up most strongly near the middle
                # of the detuning sweep and for longer buffer durations, which
                # creates a clear optimum on the 2D map without needing special
                # handling in the downstream analysis.
                separation = detuning_weight * buffer_weight
                mu_s = (y_s_ref * separation * c_ax, y_s_ref * separation * s_ax)
                mu_t = (y_t_ref * separation * c_ax, y_t_ref * separation * s_ax)

                sigma_scale = max(0.55, 1.15 - 0.35 * buffer_weight)
                params = SimulationParamsIQ(
                    n_samples=num_shots,
                    p_triplet=0.5,
                    mu_S=mu_s,
                    mu_T=mu_t,
                    sigma_I=sigma_i_base * sigma_scale,
                    sigma_Q=sigma_q_base * sigma_scale,
                    rho=0.0,
                    tau_M=tau_m,
                    T1=t1,
                )
                samples, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
                i_arr[pair_index, :, detuning_index, buffer_index] = samples[:, 0]
                q_arr[pair_index, :, detuning_index, buffer_index] = samples[:, 1]

    return xr.Dataset(
        {
            "I": (["qubit_pair", "n_runs", "detuning", "buffer_duration"], i_arr),
            "Q": (["qubit_pair", "n_runs", "detuning", "buffer_duration"], q_arr),
        },
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            "detuning": xr.DataArray(detuning_array, dims="detuning", attrs={"long_name": "detuning", "units": "V"}),
            "buffer_duration": xr.DataArray(
                buffer_ns_array.astype(float),
                dims="buffer_duration",
                attrs={"long_name": "buffer duration", "units": "ns"},
            ),
        },
    )
