from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.experiment import (
    _make_batchable_list_from_multiplexed,
)


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate simulated IQ charge-sensing data and populate the node namespace.

    Produces (1,1) and (0,2) IQ blobs whose separation grows linearly with
    integration time while noise grows as sqrt(t), giving SNR proportional to
    t_int — the expected behaviour for coherent charge sensing with
    ``measure_accumulated``.

    The returned :class:`xr.Dataset` has the same variable layout as real OPX
    data so the downstream analysis and plotting pipeline runs unchanged.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        The function writes ``quantum_dot_pairs`` and ``all_sensors`` into
        ``node.namespace``.
    """
    # -- Set up namespace (mirrors create_qua_program) -----------------------
    quantum_dots = [node.machine.get_component(k) for k in node.parameters.quantum_dots]
    if len(quantum_dots) < 2:
        raise ValueError(
            f"At least 2 Quantum Dots required. Received {len(quantum_dots)}"
        )

    quantum_dot_pair_names = [
        pair
        for dot1, dot2 in combinations(node.parameters.quantum_dots, 2)
        if (pair := node.machine.find_quantum_dot_pair(dot1, dot2)) is not None
    ]
    node.log(
        f"[sim] Found {len(quantum_dot_pair_names)} quantum dot pairs: "
        f"{quantum_dot_pair_names}"
    )
    node.namespace["quantum_dot_pairs"] = quantum_dot_pairs = [
        node.machine.get_component(k) for k in quantum_dot_pair_names
    ]
    node.namespace["all_sensors"] = all_sensors = {
        pair.name: _make_batchable_list_from_multiplexed(
            pair.sensor_dots, node.parameters.multiplexed
        )
        for pair in quantum_dot_pairs
    }

    # -- Integration time axis (same construction as create_qua_program) -----
    samples_per_chunk = node.parameters.integration_time_step // 4
    n_times = len(
        np.arange(
            node.parameters.integration_time_start,
            node.parameters.integration_time_stop,
            node.parameters.integration_time_step,
        )
    )
    time_axis = np.arange(1, n_times + 1) * samples_per_chunk * 4
    n_reps = node.parameters.num_shots

    # -- Physics model -------------------------------------------------------
    # Accumulated mean = rate * t_int ;  noise std = noise_rate * sqrt(t_int)
    # => SNR(t) = delta_rate^2 * t / (2 * noise_rate^2)
    #
    # With the values below, SNR ≈ 6.4e-3 * t_int(ns), so the default
    # threshold_SNR = 10 is crossed around t ≈ 1560 ns.
    #
    # T1 relaxation model (t1_relaxation_fraction > 0):
    #   A fraction of (0,2) shots relax to (1,1) mid-measurement.  These land
    #   near the (1,1) position, making the (0,2) blob elongated along the
    #   (0,2) → (1,1) axis.  Set t1_relaxation_fraction ∈ (0, 1) to exercise
    #   the double-Gaussian branch of the analysis.
    signal_rate_11 = np.array([1.0e-6, 2.5e-6])  # (I, Q) V per ns — (1,1) state
    signal_rate_02 = np.array([3.0e-6, 0.5e-6])  # (0,2) state
    noise_rate = 2.5e-5  # V per sqrt(ns)

    # Fraction of (0,2) shots that relax to (1,1) during readout (T1-limited).
    # Increase toward 0.4 to trigger the double-Gaussian path in the analysis.
    t1_relaxation_fraction = getattr(node.parameters, "t1_relaxation_fraction", 0.35)

    rng = np.random.default_rng(seed=42)

    # -- Generate IQ data ----------------------------------------------------
    data_vars: dict[str, xr.DataArray] = {}
    coords = {
        "repetition": np.arange(n_reps),
        "integration_time": time_axis,
    }
    dims = ["repetition", "integration_time"]

    pair_idx = 0
    for dp in quantum_dot_pairs:
        for sensor in all_sensors[dp.name]:
            key = f"{dp.name}_{sensor.name}"

            # Small per-sensor perturbation so different pairs aren't identical
            scale = 1.0 + 0.1 * pair_idx
            r_11 = signal_rate_11 * scale
            r_02 = signal_rate_02 * scale

            # (1,1) shots: straightforward Gaussian blobs at each integration time
            mean_11 = r_11[:, np.newaxis] * time_axis[np.newaxis, :]  # (2, n_times)
            std = noise_rate * np.sqrt(time_axis)[np.newaxis, :]  # (1, n_times)

            I_11 = rng.normal(loc=mean_11[0], scale=std, size=(n_reps, n_times))
            Q_11 = rng.normal(loc=mean_11[1], scale=std, size=(n_reps, n_times))

            # (0,2) shots: mix of "pure (0,2)" and "T1-relaxed → (1,1)" shots.
            # Relaxed shots are drawn from the (1,1) distribution.
            mean_02 = r_02[:, np.newaxis] * time_axis[np.newaxis, :]
            I_02_pure = rng.normal(loc=mean_02[0], scale=std, size=(n_reps, n_times))
            Q_02_pure = rng.normal(loc=mean_02[1], scale=std, size=(n_reps, n_times))
            I_02_relax = rng.normal(loc=mean_11[0], scale=std, size=(n_reps, n_times))
            Q_02_relax = rng.normal(loc=mean_11[1], scale=std, size=(n_reps, n_times))

            # Boolean mask: which shots relaxed?
            relaxed = rng.random(size=(n_reps, 1)) < t1_relaxation_fraction
            I_02 = np.where(relaxed, I_02_relax, I_02_pure)
            Q_02 = np.where(relaxed, Q_02_relax, Q_02_pure)

            for var_name, arr in [
                (f"I_11_{key}", I_11),
                (f"Q_11_{key}", Q_11),
                (f"I_02_{key}", I_02),
                (f"Q_02_{key}", Q_02),
            ]:
                data_vars[var_name] = xr.DataArray(arr, dims=dims, coords=coords)

            pair_idx += 1

    return xr.Dataset(data_vars)
