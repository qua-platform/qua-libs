from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _resolve_qubit_pairs(node: QualibrationNode):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if "qubit_pairs" in node.namespace:
        return node.namespace["qubit_pairs"]

    qubit_pairs_param = getattr(node.parameters, "qubit_pairs", None)
    machine_pairs = getattr(node.machine, "qubit_pairs", None)
    if machine_pairs is None:
        raise ValueError("Expected node.machine.qubit_pairs to be available for simulation.")

    if qubit_pairs_param not in (None, ""):
        qubit_pairs = [machine_pairs[name] for name in qubit_pairs_param]
    else:
        qubit_pairs = list(machine_pairs.values())

    node.namespace["qubit_pairs"] = qubit_pairs
    return qubit_pairs


def _build_ramp_duration_array(node: QualibrationNode) -> np.ndarray:
    """Build the ramp-duration sweep array exactly as node 07 does."""
    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)

    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. "
            f"Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    if bool(getattr(node.parameters, "ramp_log_scale", False)):
        n_ramp_pts = int((ramp_max - ramp_min) // ramp_step)
        ramp_duration_array = np.logspace(
            ramp_min,
            ramp_max,
            n_ramp_pts,
            dtype=int,
            endpoint=True,
        )
    else:
        ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)

    if ramp_duration_array.size < 1:
        raise ValueError(
            "Ramp duration sweep is empty. "
            f"Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    return ramp_duration_array


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate a simulated raw dataset for node 07 (init ramp-rate calibration).

    The output is shaped to match the *real* acquisition pipeline for this node:
    per-qubit-pair variables named ``state_{qp}``, ``I_{qp}``, and ``Q_{qp}`` with
    dimensions ``(shot, ramp_duration)`` and a ``ramp_duration`` coordinate.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``qubit_pairs`` and ``sweep_axes`` into ``node.namespace``.
    """
    qubit_pairs = _resolve_qubit_pairs(node)
    qp_names = [qp.name for qp in qubit_pairs]

    ramp_duration_array = _build_ramp_duration_array(node)
    n_ramp = int(ramp_duration_array.size)

    n_shots = int(node.parameters.num_shots)
    if n_shots < 1:
        raise ValueError(f"num_shots must be >= 1. Got {n_shots}")

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qp_names),
        "shot": xr.DataArray(np.arange(n_shots)),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    coords = {
        "shot": np.arange(n_shots),
        "ramp_duration": ramp_duration_array,
    }

    rng = np.random.default_rng(seed=42)
    find_minimum = bool(getattr(node.parameters, "find_minimum", True))

    ramp_vals = ramp_duration_array.astype(float)
    ramp_span = float(ramp_vals.max() - ramp_vals.min()) if n_ramp > 1 else 1.0

    data_vars: dict[str, xr.DataArray] = {}

    for idx, qp_name in enumerate(qp_names):
        # Pick a seeded optimum inside the sweep range.
        if n_ramp >= 6:
            low = max(1, n_ramp // 6)
            high = min(n_ramp - 1, 5 * n_ramp // 6)
            opt_idx = int(rng.integers(low=low, high=high))
        else:
            opt_idx = int(rng.integers(low=0, high=n_ramp))
        optimum = float(ramp_vals[opt_idx])

        # Width chosen to produce a clear, smooth extremum.
        sigma = max(1.0, 0.18 * ramp_span)

        p_floor = rng.uniform(0.03, 0.12)
        p_amp = rng.uniform(0.65, 0.9)
        p_amp = min(p_amp, 0.98 - p_floor)

        if find_minimum:
            # Minimum at optimum, higher away from optimum.
            p_state = p_floor + p_amp * (1.0 - np.exp(-((ramp_vals - optimum) / sigma) ** 2))
        else:
            # Maximum at optimum, lower away from optimum.
            p_state = p_floor + p_amp * np.exp(-((ramp_vals - optimum) / sigma) ** 2)

        p_state = np.clip(p_state, 1e-3, 1.0 - 1e-3)

        # Bernoulli shots per ramp duration.
        state = (rng.random((n_shots, n_ramp)) < p_state[None, :]).astype(int)

        # Simulated I/Q readout clusters, separated by state with mild drift vs ramp.
        phase = 0.6 + 0.3 * idx
        drift = 0.02 * np.sin(2.0 * np.pi * (ramp_vals - ramp_vals.min()) / (ramp_span + 1e-12) + phase)

        i0 = -0.25 + drift
        i1 = 0.25 + drift
        q0 = 0.05 * np.cos(phase) + 0.5 * drift
        q1 = 0.35 + 0.05 * np.sin(phase) + 0.5 * drift

        noise_i = rng.uniform(0.03, 0.06)
        noise_q = rng.uniform(0.03, 0.06)

        I = np.where(state == 0, i0[None, :], i1[None, :]) + rng.normal(0.0, noise_i, size=(n_shots, n_ramp))
        Q = np.where(state == 0, q0[None, :], q1[None, :]) + rng.normal(0.0, noise_q, size=(n_shots, n_ramp))

        data_vars[f"state_{qp_name}"] = xr.DataArray(
            state,
            dims=("shot", "ramp_duration"),
            coords=coords,
            attrs={"long_name": "State assignment", "units": "arb."},
        )
        data_vars[f"I_{qp_name}"] = xr.DataArray(
            I,
            dims=("shot", "ramp_duration"),
            coords=coords,
            attrs={"long_name": "I quadrature", "units": "arb."},
        )
        data_vars[f"Q_{qp_name}"] = xr.DataArray(
            Q,
            dims=("shot", "ramp_duration"),
            coords=coords,
            attrs={"long_name": "Q quadrature", "units": "arb."},
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "shot": xr.DataArray(
                np.arange(n_shots),
                dims=("shot",),
                attrs={"long_name": "shot"},
            ),
            "ramp_duration": xr.DataArray(
                ramp_duration_array,
                dims=("ramp_duration",),
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
            "qubit_pair": xr.DataArray(
                qp_names,
                dims=("qubit_pair",),
                attrs={"long_name": "qubit pair"},
            ),
        },
        attrs={"source": "simulated", "node": "07a_init_ramp_rate_calibration"},
    )

