from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from calibration_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qualibrate.core import QualibrationNode


def _resolve_qubit_pairs(node: "QualibrationNode") -> List:
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def _duration_unit(t_ns: float, t_min: float, t_max: float) -> float:
    span = t_max - t_min
    if abs(span) < 1e-15:
        return 0.5
    return float(np.clip((t_ns - t_min) / span, 0.0, 1.0))


def build_ramp_duration_sweep(ramp_duration_min: int, ramp_duration_max: int, ramp_duration_step: int) -> np.ndarray:
    """Build ramp duration grid (ns), same rules as 06d (multiples of 4 validated by caller)."""
    r_min = int(ramp_duration_min)
    r_max = int(ramp_duration_max)
    step = int(ramp_duration_step)
    return np.arange(r_min, r_max, step, dtype=int)


def generate_simulated_dataset(node: "QualibrationNode") -> xr.Dataset:
    """Synthetic PSB ramp-duration sweep; contrast improves toward longer ramps (toy model)."""
    qubit_pairs = _resolve_qubit_pairs(node)
    pair_names = [qp.name for qp in qubit_pairs]

    ramp_array = build_ramp_duration_sweep(
        node.parameters.ramp_duration_min,
        node.parameters.ramp_duration_max,
        node.parameters.ramp_duration_step,
    )
    if len(ramp_array) == 0:
        raise ValueError(
            "Empty ramp duration sweep: check ramp_duration_min < ramp_duration_max with positive step."
        )

    sweep_name = node.parameters.sweep_name
    ramp_floats = ramp_array.astype(float)
    d_min = float(ramp_floats.min())
    d_max = float(ramp_floats.max())

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in qubit_pairs]
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(pair_names),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        sweep_name: xr.DataArray(
            ramp_floats,
            attrs={"long_name": "ramp duration", "units": "ns"},
        ),
    }

    num_shots = int(node.parameters.num_shots)
    n_pairs = len(qubit_pairs)
    n_d = len(ramp_array)

    I_arr = np.zeros((n_pairs, num_shots, n_d), dtype=float)
    Q_arr = np.zeros((n_pairs, num_shots, n_d), dtype=float)

    tau_M = 1.0
    T1 = 2.0
    sigma_I_base = 0.12e-2
    sigma_Q_base = 0.10e-2

    for pi, _qp in enumerate(qubit_pairs):
        rng = np.random.default_rng(seed=42 + pi * 9973)
        theta = 0.38 + 0.27 * float(pi)
        c_ax, s_ax = float(np.cos(theta)), float(np.sin(theta))
        y_s = (-1.2e-2) * (1.0 + 0.06 * float(pi))
        y_t = (1.2e-2) * (1.0 + 0.06 * float(pi))
        mu_s = (y_s * c_ax, y_s * s_ax)
        mu_t = (y_t * c_ax, y_t * s_ax)

        for di, d_ns in enumerate(ramp_floats):
            u = _duration_unit(float(d_ns), d_min, d_max)
            # Longer ramp -> better settling -> larger separation (toy).
            sep = 0.2 + 0.8 * u
            mu_s_t = (mu_s[0] * sep, mu_s[1] * sep)
            mu_t_t = (mu_t[0] * sep, mu_t[1] * sep)
            scale = float(np.sqrt(max(float(d_ns) / max(d_max, 1.0), 1e-6)))
            sigma_I = sigma_I_base / scale
            sigma_Q = sigma_Q_base / scale
            params = SimulationParamsIQ(
                n_samples=num_shots,
                p_triplet=0.5,
                mu_S=mu_s_t,
                mu_T=mu_t_t,
                sigma_I=sigma_I,
                sigma_Q=sigma_Q,
                rho=0.0,
                tau_M=tau_M,
                T1=T1,
            )
            X, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
            I_arr[pi, :, di] = X[:, 0]
            Q_arr[pi, :, di] = X[:, 1]

    dims = ("qubit_pair", "n_runs", sweep_name)
    return xr.Dataset(
        {"I": (dims, I_arr), "Q": (dims, Q_arr)},
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            sweep_name: xr.DataArray(
                ramp_floats,
                dims=sweep_name,
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
        },
    )


def plot_simulated_dataset_histograms(
    ds: xr.Dataset,
    *,
    sweep_name: str = "ramp_duration",
    qubit_pairs: Optional[Sequence[Union[str, object]]] = None,
    n_bins: int = 48,
    log_counts: bool = True,
    edge_fraction: float = 1.0 / 3.0,
) -> "Figure":
    """Delegate to measure-duration helper (generic on sweep_name)."""
    from calibration_utils.psb_search_sweep_measure_duration.simulated_data_generator import (
        plot_simulated_dataset_histograms as _plot_hist,
    )

    return _plot_hist(
        ds,
        sweep_name=sweep_name,
        qubit_pairs=qubit_pairs,
        n_bins=n_bins,
        log_counts=log_counts,
        edge_fraction=edge_fraction,
    )
