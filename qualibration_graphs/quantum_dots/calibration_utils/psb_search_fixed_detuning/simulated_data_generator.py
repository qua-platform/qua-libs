from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from calibration_utils.iq_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)
from calibration_utils.psb_search_sweep_detuning.simulated_data_generator import (
    _psb_eye_branch_scalars,
)
from .helper_utils import resolve_qubits_and_dot_pairs

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def _simulate_state_iq(
    *,
    num_shots: int,
    mu_s: tuple[float, float],
    mu_t: tuple[float, float],
    p_triplet: float,
    sigma_I: float,
    sigma_Q: float,
    tau_M: float,
    T1: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one pure-state arm using the shared Barthel IQ forward model."""

    params = SimulationParamsIQ(
        n_samples=num_shots,
        p_triplet=p_triplet,
        mu_S=mu_s,
        mu_T=mu_t,
        sigma_I=sigma_I,
        sigma_Q=sigma_Q,
        rho=0.0,
        tau_M=tau_M,
        T1=T1,
    )
    X, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
    return X[:, 0], X[:, 1]


def generate_simulated_dataset(node: "QualibrationNode") -> xr.Dataset:
    """Synthetic labeled two-arm PSB IQ that matches the real 06d ``ds_raw`` contract."""

    qubits, qubit_dot_pairs = resolve_qubits_and_dot_pairs(node)
    qnames = [qubit.name for qubit in qubits]
    num_shots = int(node.parameters.num_shots)

    node.namespace["qubits"] = qubits
    node.namespace["qubit_dot_pairs"] = qubit_dot_pairs
    node.namespace["tracked_original_detunings"] = {}
    node.namespace["sweep_axes"] = {
        "n_runs": xr.DataArray(np.arange(num_shots), attrs={"long_name": "shot"}),
    }

    I_no_pi = np.zeros((len(qubits), num_shots), dtype=float)
    Q_no_pi = np.zeros((len(qubits), num_shots), dtype=float)
    I_pi = np.zeros((len(qubits), num_shots), dtype=float)
    Q_pi = np.zeros((len(qubits), num_shots), dtype=float)

    tau_M = 1.0
    T1 = 2.0
    sigma_I = 0.12e-2
    sigma_Q = 0.10e-2
    t_mid = 0.5

    for idx, _qubit in enumerate(qubits):
        theta = 0.38 + 0.27 * float(idx)
        c_ax, s_ax = float(np.cos(theta)), float(np.sin(theta))
        y_s_ref = (-0.5e-2) * (1.0 + 0.06 * float(idx))
        y_t_ref = (0.5e-2) * (1.0 + 0.06 * float(idx))
        y_left = float(np.clip((-1.15e-2) * (1.0 + 0.12 * float(idx)), y_s_ref + 1e-12, y_t_ref - 1e-12))
        y_right = float(np.clip((1.05e-2) * (1.0 + 0.12 * float(idx)), y_s_ref + 1e-12, y_t_ref - 1e-12))

        y_s, y_t = _psb_eye_branch_scalars(
            t_mid,
            y_left=y_left,
            y_right=y_right,
            y_s_ref=y_s_ref,
            y_t_ref=y_t_ref,
        )
        mu_s = (y_s * c_ax, y_s * s_ax)
        mu_t = (y_t * c_ax, y_t * s_ax)

        no_pi_is_triplet = node.parameters.init_state_label == "decay"
        p_triplet_no_pi = 1.0 if no_pi_is_triplet else 0.0
        p_triplet_pi = 0.0 if no_pi_is_triplet else 1.0

        I_no_pi[idx], Q_no_pi[idx] = _simulate_state_iq(
            num_shots=num_shots,
            mu_s=mu_s,
            mu_t=mu_t,
            p_triplet=p_triplet_no_pi,
            sigma_I=sigma_I,
            sigma_Q=sigma_Q,
            tau_M=tau_M,
            T1=T1,
            rng=np.random.default_rng(seed=42_001 + idx * 9973),
        )
        I_pi[idx], Q_pi[idx] = _simulate_state_iq(
            num_shots=num_shots,
            mu_s=mu_s,
            mu_t=mu_t,
            p_triplet=p_triplet_pi,
            sigma_I=sigma_I,
            sigma_Q=sigma_Q,
            tau_M=tau_M,
            T1=T1,
            rng=np.random.default_rng(seed=52_001 + idx * 9973),
        )

    return xr.Dataset(
        {
            "I_no_pi": (["qubit", "n_runs"], I_no_pi),
            "Q_no_pi": (["qubit", "n_runs"], Q_no_pi),
            "I_pi": (["qubit", "n_runs"], I_pi),
            "Q_pi": (["qubit", "n_runs"], Q_pi),
        },
        coords={
            "qubit": qnames,
            "n_runs": np.arange(num_shots),
        },
    )
