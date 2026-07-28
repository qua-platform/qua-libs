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


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate a simulated raw dataset for the init-2D calibration (ramp × wait).

    The real QUA program averages over shots on the OPX, so the returned variables
    match that shape: 2D arrays indexed by ``(ramp_duration, wait_duration)``.

    Parameters
    ----------
    node : QualibrationNode
        Calibration node whose ``parameters`` and ``machine`` are already set.
        Writes ``qubit_pairs`` and ``sweep_axes`` into ``node.namespace``.
    """
    qubit_pairs = _resolve_qubit_pairs(node)
    qp_names = [qp.name for qp in qubit_pairs]

    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)
    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. "
            f"Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    wait_min = int(node.parameters.wait_duration_min)
    wait_max = int(node.parameters.wait_duration_max)
    wait_step = int(node.parameters.wait_duration_step)
    if wait_min % 4 != 0 or wait_max % 4 != 0 or wait_step % 4 != 0:
        raise ValueError(
            f"Wait settings must be divisible by 4. "
            f"Got min={wait_min}, max={wait_max}, step={wait_step}"
        )
    if wait_min < 16:
        raise ValueError(
            f"Minimum wait duration must be >= 16 ns (4 clock cycles). Got {wait_min}"
        )

    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)
    wait_duration_array = np.arange(wait_min, wait_max, wait_step, dtype=int)

    if ramp_duration_array.size < 1 or wait_duration_array.size < 1:
        raise ValueError("Empty sweep axis for simulated init_2d dataset.")

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qp_names),
        "ramp_duration": xr.DataArray(
            ramp_duration_array,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
        "wait_duration": xr.DataArray(
            wait_duration_array,
            attrs={"long_name": "wait duration", "units": "ns"},
        ),
    }

    ramp = ramp_duration_array.astype(float)
    wait = wait_duration_array.astype(float)
    rr, ww = np.meshgrid(ramp, wait, indexing="ij")  # (n_ramp, n_wait)

    rng = np.random.default_rng(seed=42)
    find_minimum = bool(getattr(node.parameters, "find_minimum", True))

    data_vars: dict[str, xr.DataArray] = {}

    for idx, qp_name in enumerate(qp_names):
        # Choose an "optimum" inside the sweep range.
        r0 = float(rng.choice(ramp))
        w0 = float(rng.choice(wait))

        r_sigma = max(20.0, 0.18 * float(ramp.max() - ramp.min() if ramp.size > 1 else 1.0))
        w_sigma = max(20.0, 0.22 * float(wait.max() - wait.min() if wait.size > 1 else 1.0))

        # A smooth basin/peak around (r0, w0).
        gauss = np.exp(-((rr - r0) / r_sigma) ** 2 - ((ww - w0) / w_sigma) ** 2)

        # Add an oscillatory component along the wait axis so the FFT panels are meaningful.
        period_ns = rng.uniform(120.0, 450.0)
        phase = 0.4 * idx + rng.uniform(-0.5, 0.5)
        oscill = np.sin(2.0 * np.pi * ww / period_ns + phase)
        osc_weight = 0.08 + 0.02 * idx

        if find_minimum:
            state = 0.15 + 0.75 * (1.0 - gauss) + osc_weight * oscill
        else:
            state = 0.15 + 0.75 * gauss + osc_weight * oscill

        state = np.clip(state, 0.0, 1.0)

        # I/Q average signals correlated with state + a small independent oscillation.
        i_base = -0.2 + 0.6 * (1.0 - state) + 0.04 * np.cos(2.0 * np.pi * ww / (0.7 * period_ns) + phase)
        q_base = 0.15 + 0.4 * state + 0.03 * np.sin(2.0 * np.pi * ww / (0.9 * period_ns) + phase)

        i_noise = rng.normal(0.0, 0.01, size=state.shape)
        q_noise = rng.normal(0.0, 0.01, size=state.shape)

        data_vars[f"state_{qp_name}"] = xr.DataArray(
            state,
            dims=("ramp_duration", "wait_duration"),
            coords={"ramp_duration": ramp_duration_array, "wait_duration": wait_duration_array},
        )
        data_vars[f"I_{qp_name}"] = xr.DataArray(
            i_base + i_noise,
            dims=("ramp_duration", "wait_duration"),
            coords={"ramp_duration": ramp_duration_array, "wait_duration": wait_duration_array},
        )
        data_vars[f"Q_{qp_name}"] = xr.DataArray(
            q_base + q_noise,
            dims=("ramp_duration", "wait_duration"),
            coords={"ramp_duration": ramp_duration_array, "wait_duration": wait_duration_array},
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "ramp_duration": xr.DataArray(
                ramp_duration_array,
                dims=("ramp_duration",),
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
            "wait_duration": xr.DataArray(
                wait_duration_array,
                dims=("wait_duration",),
                attrs={"long_name": "wait duration", "units": "ns"},
            ),
            "qubit_pair": xr.DataArray(
                qp_names,
                dims=("qubit_pair",),
                attrs={"long_name": "qubit pair"},
            ),
        },
        attrs={"source": "simulated", "node": "init_2d"},
    )

