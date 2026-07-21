"""Single-qubit randomized benchmarking analysis — exponential decay fit.

The survival probability at circuit depth *m* follows

    F(m) = A · α^m + B

where α is the depolarising parameter.  Per-Clifford and per-native-gate
metrics are derived as follows (d = 2 for a single qubit):

    epc  = (1 − α) · (d−1)/d          # average error per Clifford
    F_cliff = 1 − epc

    α_gate = α^(1/⟨n_g⟩)             # depolarising param per physical gate
    epg  = (1 − α_gate) · (d−1)/d    # average error per native gate
    F_gate = 1 − epg

⟨n_g⟩ is the average number of physical (non-Z) gates per Clifford, which
differs between decompositions:

    _DECOMPOSITION_SEQUENCES              ⟨n_g⟩ = 20/24 ≈ 0.833
    _ALTERNATIVE_DECOMPOSITION_SEQUENCES  ⟨n_g⟩ = 44/24 ≈ 1.833

The α_gate = α^(1/⟨n_g⟩) formula assumes ⟨α_gate^{n_k}⟩ ≈ α_gate^{⟨n_k⟩},
valid when gate errors are small (correction ~ Var(n_k)·(log α_gate)²/2).

Use ``clifford_tables.avg_physical_gates_per_clifford(sequences)`` to compute
this value for any decomposition and pass it to ``fit_raw_data``.

The fit uses ``scipy.optimize.curve_fit`` with physical bounds on A, α, B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

import xarray as xr

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fit result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fitted parameters for a single qubit's RB decay."""

    alpha: float = 0.0
    """Depolarising parameter (decay constant per Clifford)."""
    A: float = 0.0
    """Amplitude of the exponential decay."""
    B: float = 0.0
    """Asymptotic offset (ideally 0.5 for single qubit)."""
    error_per_clifford: float = 0.0
    """Average error per Clifford gate:  epc = (1−α)/2."""
    clifford_fidelity: float = 0.0
    """Average Clifford gate fidelity (1 − epc)."""
    avg_gates_per_clifford: float = 0.0
    """Average number of physical (non-Z) gates per Clifford used in the experiment."""
    error_per_gate: float = 0.0
    """Average error per native gate:  epg = (1 − α^(1/⟨n_g⟩))/2."""
    native_gate_fidelity: float = 0.0
    """Average native gate fidelity (1 − epg)."""
    success: bool = False
    """Whether the fit converged to a physically sensible result."""


# ──────────────────────────────────────────────────────────────────────────────
# Internal fitting
# ──────────────────────────────────────────────────────────────────────────────


def _rb_decay(m: np.ndarray, A: float, alpha: float, B: float) -> np.ndarray:
    """Exponential decay model: F(m) = A · α^m + B."""
    return A * alpha**m + B


def _fit_single_qubit(
    depths: np.ndarray,
    survival_prob: np.ndarray,
    avg_gates_per_clifford: float,
) -> dict[str, Any]:
    """Fit survival probability vs circuit depth to extract gate fidelity.

    Parameters
    ----------
    depths : 1-D array
        Circuit depths (number of Cliffords).
    survival_prob : 1-D array
        Average survival probability at each depth.
    avg_gates_per_clifford : float
        Average number of physical (non-Z) gates per Clifford in the
        decomposition used by the experiment.  Used to convert epc → epg.

    Returns
    -------
    dict
        Keys: ``alpha``, ``A``, ``B``, ``error_per_clifford``,
        ``clifford_fidelity``, ``avg_gates_per_clifford``,
        ``error_per_gate``, ``native_gate_fidelity``,
        ``fitted_curve``, ``success``.
    """
    depths = np.asarray(depths, dtype=np.float64)
    y = np.asarray(survival_prob, dtype=np.float64)

    if len(depths) < 3 or len(y) < 3:
        return dict(
            FitParameters().__dict__,
            fitted_curve=np.full_like(y, np.nan),
        )

    # try:
    popt, _ = curve_fit(
        _rb_decay,
        depths,
        y,
        p0=[0.0, 0.99, 0.5],
        bounds=([-1, 0, 0], [1, 1, 1]),
        maxfev=10_000,
    )
    A, alpha, B = popt
    success = bool(np.isfinite(alpha) and 0 < alpha <= 1)
    # except (RuntimeError, ValueError):
    #     A, alpha, B = 0.5, 0.5, 0.5
    #     success = False

    d = 2  # single qubit
    epc = (1.0 - alpha) * (d - 1) / d

    # α = α_gate^⟨n_g⟩  →  α_gate = α^(1/⟨n_g⟩)
    alpha_gate = float(alpha) ** (1.0 / avg_gates_per_clifford)
    epg = (1.0 - alpha_gate) * (d - 1) / d

    fitted_curve = _rb_decay(depths, A, alpha, B)

    return {
        "alpha": float(alpha),
        "A": float(A),
        "B": float(B),
        "error_per_clifford": float(epc),
        "clifford_fidelity": float(1.0 - epc),
        "avg_gates_per_clifford": avg_gates_per_clifford,
        "error_per_gate": float(epg),
        "native_gate_fidelity": float(1.0 - epg),
        "fitted_curve": fitted_curve,
        "success": success,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def _get_qubit_state_data(ds_raw: xr.Dataset, qname: str) -> np.ndarray | None:
    """Extract per-qubit state data, handling both naming conventions.

    The ``XarrayDataFetcher`` regex groups ``state_q1``, ``state_q2`` into a
    single ``state_q`` variable stacked along the ``qubit`` dimension.  This
    helper checks for a per-qubit variable first, then falls back to a stacked
    variable with a ``qubit`` dimension.
    """
    var_name = f"state_{qname}"
    if var_name in ds_raw.data_vars:
        return ds_raw[var_name].values
    for candidate in ds_raw.data_vars:
        da = ds_raw[candidate]
        if candidate.startswith("state") and "qubit" in da.dims:
            try:
                return da.sel(qubit=qname).values
            except (KeyError, ValueError):
                continue
    return None


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    avg_gates_per_clifford: float,
) -> dict[str, dict[str, Any]]:
    """Run the RB exponential-decay fit for every qubit.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with coordinates ``depth`` and ``circuit``, and data
        variables ``state_<qubit_name>`` shaped ``[num_circuits, num_depths]``.
    qubits : list
        Qubit objects (each must have a ``.name`` attribute).
    avg_gates_per_clifford : float
        Average number of physical (non-Z) gates per Clifford for the
        decomposition used in the experiment.  Pass the value from
        ``clifford_tables.avg_physical_gates_per_clifford(sequences)``.

    Returns
    -------
    dict
        ``{qubit_name: {alpha, A, B, error_per_clifford, clifford_fidelity,
        avg_gates_per_clifford, error_per_gate, native_gate_fidelity,
        fitted_curve, success}}``.
    """
    depths = ds_raw.coords["depth"].values.astype(np.float64)
    results: dict[str, dict[str, Any]] = {}

    for qubit in qubits:
        qname = qubit.name
        state_data = _get_qubit_state_data(ds_raw, qname)
        if state_data is None:
            logger.warning("No state variable for qubit %s — skipping.", qname)
            results[qname] = dict(
                FitParameters().__dict__,
                fitted_curve=np.array([]),
            )
            continue

        survival_prob = np.mean(state_data, axis=0)  # average over circuits
        results[qname] = _fit_single_qubit(depths, survival_prob, avg_gates_per_clifford)

    return results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    node_logger: Any | None = None,
) -> None:
    """Log fitted RB results for all qubits.

    Parameters
    ----------
    fit_results : dict
        Output of :func:`fit_raw_data`.
    node_logger : callable, optional
        Logging function (e.g. ``node.log``).  Falls back to module logger.
    """
    _log = node_logger or logger.info
    for qname, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        msg = (
            f"  {qname}: [{status}] "
            f"F_cliff = {r['clifford_fidelity']:.4f}, epc = {r['error_per_clifford']:.5f} | "
            f"F_gate = {r['native_gate_fidelity']:.4f}, epg = {r['error_per_gate']:.5f} "
            f"(⟨n_g⟩ = {r['avg_gates_per_clifford']:.3f}) | "
            f"α = {r['alpha']:.5f}"
        )
        _log(msg)
