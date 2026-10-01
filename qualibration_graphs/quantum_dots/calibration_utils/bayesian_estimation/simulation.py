"""Pure-Python synthetic Bayesian Ramsey data matching the QUA update in 20a_BayesianEstimation."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import xarray as xr

from calibration_utils.bayesian_estimation.idle_grid import sweep_from_parameters
from calibration_utils.bayesian_estimation.parameters import Parameters


def simulate_bayesian_ramsey_dataset(
    p: Parameters,
    f_true_mhz: float,
    qubit_names: Sequence[str],
    *,
    seed: int | None = None,
) -> xr.Dataset:
    """Simulate binary ``state`` streams and posterior ``Pf`` consistent with 20a QUA logic.

    For each repetition (shot), ``Pf`` starts uniform, is updated sequentially at each idle
    time, then reset to uniform before the next shot. Readouts are Bernoulli with
    P(1|τ) = 0.5·(1 + α + β·cos(2π f_true τ_us)) with τ_us = τ_ns·10⁻³, clipped for stability.

    Each qubit gets an independent noise realization; grids and (α, β) come from ``p``.
    """
    names: List[str] = list(qubit_names)
    if not names:
        raise ValueError("qubit_names must be non-empty.")

    v_f, tau_ns, _, _ = sweep_from_parameters(p)
    if v_f.size == 0 or tau_ns.size == 0:
        raise ValueError("Empty frequency or tau grid from parameters.")

    n_shots = int(p.num_shots)
    n_tau = int(tau_ns.size)
    n_f = int(v_f.size)
    alpha = float(p.alpha)
    beta = float(p.beta)

    rng = np.random.default_rng(seed)
    tau_us = np.asarray(tau_ns, dtype=float) * 1e-3

    n_q = len(names)
    pf_out = np.zeros((n_q, n_shots, n_tau, n_f), dtype=np.float64)
    state_out = np.zeros((n_q, n_shots, n_tau), dtype=np.int32)

    for qi in range(n_q):
        for shot in range(n_shots):
            pf = np.ones(n_f, dtype=np.float64) / n_f
            for j in range(n_tau):
                c_true = np.cos(2.0 * np.pi * f_true_mhz * tau_us[j])
                p1 = 0.5 * (1.0 + alpha + beta * c_true)
                p1 = float(np.clip(p1, 1e-9, 1.0 - 1e-9))
                state = int(rng.binomial(1, p1))
                state_out[qi, shot, j] = state
                rk = state - 0.5
                c_grid = np.cos(2.0 * np.pi * v_f * tau_us[j])
                lik = 0.5 + rk * (alpha + beta * c_grid)
                pf = pf * lik
                s = pf.sum()
                if s <= 0.0 or not np.isfinite(s):
                    pf = np.ones(n_f, dtype=np.float64) / n_f
                else:
                    pf = pf / s
                pf_out[qi, shot, j, :] = pf

    tau_coord = xr.DataArray(
        np.asarray(tau_ns, dtype=float),
        dims="tau",
        attrs={"long_name": "idle time", "units": "ns"},
    )
    f_coord = xr.DataArray(
        np.asarray(v_f, dtype=float),
        dims="frequency",
        attrs={"long_name": "hypothesis frequency", "units": "MHz"},
    )
    coords = {
        "repetition": np.arange(n_shots, dtype=int),
        "tau": tau_coord,
        "frequency": f_coord,
    }

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    for i, _name in enumerate(names, start=1):
        data_vars[f"Pf{i}"] = (["repetition", "tau", "frequency"], pf_out[i - 1])
        data_vars[f"state{i}"] = (["repetition", "tau"], state_out[i - 1])

    return xr.Dataset(data_vars, coords=coords)
