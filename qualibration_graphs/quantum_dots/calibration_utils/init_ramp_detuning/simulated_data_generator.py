from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from calibration_utils.init_ramp_rate.helper_utils import validate_and_build_ramp_sweep

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


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate a simulated raw dataset for the init 2D (ramp × detuning) calibration.

    The real QUA program averages over shots on the OPX, so the returned variables
    match that shape: ``state``, ``I``, and ``Q`` arrays indexed by
    ``(qubit_pair, ramp_duration, detuning)``.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``qubit_pairs`` and ``sweep_axes`` into ``node.namespace``.
    """
    qubit_pairs = _resolve_qubit_pairs(node)
    qp_names = [qp.name for qp in qubit_pairs]

    detuning_min = float(node.parameters.detuning_min)
    detuning_max = float(node.parameters.detuning_max)
    detuning_step = float(node.parameters.detuning_step)

    ramp_duration_array = validate_and_build_ramp_sweep(node)
    detuning_array = np.arange(detuning_min, detuning_max, detuning_step, dtype=float)

    if ramp_duration_array.size < 1 or detuning_array.size < 1:
        raise ValueError("Empty sweep axis for simulated init_ramp_detuning dataset.")

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qp_names),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "detuning": xr.DataArray(
            detuning_array,
            attrs={"long_name": "detuning", "units": "V"},
        ),
    }

    ramp = ramp_duration_array.astype(float)
    det = detuning_array.astype(float)
    rr, dd = np.meshgrid(ramp, det, indexing="ij")  # (n_ramp, n_det)

    rng = np.random.default_rng(seed=42)
    find_minimum = bool(getattr(node.parameters, "find_minimum", True))

    state_rows = []
    i_rows = []
    q_rows = []

    for idx, qp_name in enumerate(qp_names):
        r0 = float(rng.choice(ramp))
        d0 = float(rng.choice(det))

        r_sigma = max(20.0, 0.20 * float(ramp.max() - ramp.min() if ramp.size > 1 else 1.0))
        d_sigma = max(0.01, 0.18 * float(det.max() - det.min() if det.size > 1 else 1.0))

        gauss = np.exp(-(((rr - r0) / r_sigma) ** 2) - ((dd - d0) / d_sigma) ** 2)

        # Add a mild oscillatory component along detuning so the FFT panel is non-trivial.
        period_v = rng.uniform(0.06, 0.18)
        phase = 0.4 * idx + rng.uniform(-0.5, 0.5)
        oscill = np.sin(2.0 * np.pi * (dd - det.min()) / period_v + phase)
        osc_weight = 0.05 + 0.02 * idx

        if find_minimum:
            state = 0.15 + 0.75 * (1.0 - gauss) + osc_weight * oscill
        else:
            state = 0.15 + 0.75 * gauss + osc_weight * oscill

        state = np.clip(state, 0.0, 1.0)

        i_base = -0.2 + 0.6 * (1.0 - state) + 0.03 * np.cos(2.0 * np.pi * dd / (0.7 * period_v) + phase)
        q_base = 0.15 + 0.4 * state + 0.03 * np.sin(2.0 * np.pi * dd / (0.9 * period_v) + phase)

        i_noise = rng.normal(0.0, 0.01, size=state.shape)
        q_noise = rng.normal(0.0, 0.01, size=state.shape)

        state_rows.append(state)
        i_rows.append(i_base + i_noise)
        q_rows.append(q_base + q_noise)

    coords = {
        "qubit_pair": xr.DataArray(qp_names, dims=("qubit_pair",), attrs={"long_name": "qubit pair"}),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            dims=("ramp_duration",),
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "detuning": xr.DataArray(
            detuning_array,
            dims=("detuning",),
            attrs={"long_name": "detuning", "units": "V"},
        ),
    }
    return xr.Dataset(
        {
            "state": xr.DataArray(
                np.asarray(state_rows, dtype=float),
                dims=("qubit_pair", "ramp_duration", "detuning"),
                coords=coords,
            ),
            "I": xr.DataArray(
                np.asarray(i_rows, dtype=float),
                dims=("qubit_pair", "ramp_duration", "detuning"),
                coords=coords,
            ),
            "Q": xr.DataArray(
                np.asarray(q_rows, dtype=float),
                dims=("qubit_pair", "ramp_duration", "detuning"),
                coords=coords,
            ),
        },
        attrs={"source": "simulated", "node": "init_ramp_detuning"},
    )
