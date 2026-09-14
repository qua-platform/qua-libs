"""Two-qubit gate unitary estimation via Nelder-Mead optimization.

Estimates the physical parameters (iSWAP angle, CPhase angle, single-qubit
Z rotations) of the 2Q gate by maximizing the measured XEB fidelity over
different parameterizations of the ideal 2Q unitary.

The optimization varies:
    θ_iswap  ∈ [0, π/2]   (iSWAP angle)
    φ_cphase ∈ [0, 2π]    (conditional phase)
    φ_rz1    ∈ [0, 2π]    (Z rotation on qubit 1)
    φ_rz2    ∈ [0, 2π]    (Z rotation on qubit 2)

and finds the set that yields the highest mean linear XEB fidelity across
all sequences and depths.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .ideal_probabilities import calc_ideal_probs_2q
from .analysis import calc_linear_xeb_fidelity


def _objective(
    params: np.ndarray,
    gate_indices: np.ndarray,
    depths: np.ndarray,
    measured_probs: np.ndarray,
    gate_set: Literal["sw", "t"],
) -> float:
    """Negative mean linear XEB fidelity for given 2Q parameters."""
    theta_iswap, phi_cphase, phi_rz1, phi_rz2 = params

    ideal_probs = calc_ideal_probs_2q(
        gate_indices,
        depths,
        gate_set=gate_set,
        theta_iswap=theta_iswap,
        phi_cphase=phi_cphase,
        phi_rz1=phi_rz1,
        phi_rz2=phi_rz2,
        insert_2q_gate=True,
    )

    fid = calc_linear_xeb_fidelity(measured_probs, ideal_probs, dim=4)
    return -np.mean(fid)


def estimate_2q_unitary(
    gate_indices: np.ndarray,
    depths: np.ndarray,
    measured_probs: np.ndarray,
    gate_set: Literal["sw", "t"] = "sw",
    x0: np.ndarray | None = None,
) -> dict:
    """Estimate the 2Q gate unitary from XEB data via Nelder-Mead.

    Parameters
    ----------
    gate_indices : ndarray, shape (n_sequences, 2, max_depth)
        Random gate indices for both qubits.
    depths : ndarray of int
        Circuit depths used.
    measured_probs : ndarray, shape (n_sequences, n_depths, 4)
        Measured joint probability distributions.
    gate_set : {"sw", "t"}
        Gate set identifier.
    x0 : ndarray or None
        Initial guess [θ_iswap, φ_cphase, φ_rz1, φ_rz2].
        Defaults to CZ starting point [0, π, 0, 0].

    Returns
    -------
    dict
        theta_iswap, phi_cphase, phi_rz1, phi_rz2 : fitted parameters
        fidelity : achieved mean linear XEB fidelity
        success : whether optimizer converged
        result : full scipy OptimizeResult
    """
    if x0 is None:
        x0 = np.array([0.0, np.pi, 0.0, 0.0])

    result = minimize(
        _objective,
        x0,
        args=(gate_indices, depths, measured_probs, gate_set),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6},
    )

    return {
        "theta_iswap": result.x[0],
        "phi_cphase": result.x[1],
        "phi_rz1": result.x[2],
        "phi_rz2": result.x[3],
        "fidelity": -result.fun,
        "success": result.success,
        "result": result,
    }
