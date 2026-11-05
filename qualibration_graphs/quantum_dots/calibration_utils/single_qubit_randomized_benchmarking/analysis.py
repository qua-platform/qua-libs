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

    try:
        popt, _ = curve_fit(
            _rb_decay,
            depths,
            y,
            p0=[0.0, 0.99, 0.5],
            bounds=([-1, 0, 0], [1, 1, 1]),
            maxfev=10_000,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("RB fit failed: %s", exc)
        return dict(
            FitParameters().__dict__,
            fitted_curve=np.full_like(y, np.nan),
        )

    A, alpha, B = popt
    success = bool(np.isfinite(alpha) and 0 < alpha <= 1)
    if not success:
        return dict(
            FitParameters().__dict__,
            fitted_curve=np.full_like(y, np.nan),
        )

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
    """Extract per-qubit state data, preferring the stacked ``state`` dataset."""
    if "state" in ds_raw.data_vars:
        try:
            return ds_raw.state.sel(qubit=qname, drop=True).transpose("circuit", "depth").values.astype(float)
        except (KeyError, ValueError):
            pass

    var_name = f"state_{qname}"
    if var_name in ds_raw.data_vars:
        return ds_raw[var_name].transpose("circuit", "depth").values.astype(float)
    for candidate in ds_raw.data_vars:
        da = ds_raw[candidate]
        if candidate.startswith("state") and "qubit" in da.dims:
            try:
                return da.sel(qubit=qname, drop=True).transpose("circuit", "depth").values.astype(float)
            except (KeyError, ValueError):
                continue
    return None


def process_raw_dataset(ds_raw: xr.Dataset) -> xr.Dataset:
    """Return the processed RB dataset used for fitting."""
    return ds_raw


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

    for qi, qubit in enumerate(qubits):
        qname = getattr(qubit, "name", f"q{qi}")
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


def analyse_raw_data(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    avg_gates_per_clifford: float,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Fit RB decay data and build a plotting-ready fit dataset.

    Returns
    -------
    tuple
        ``(ds_fit, fit_results)`` where ``ds_fit`` contains stacked
        per-qubit survival probabilities and fitted curves vs circuit depth.
    """
    fit_results = fit_raw_data(ds_raw, qubits, avg_gates_per_clifford)

    ds_fit = xr.Dataset(coords={"depth": ds_raw.coords["depth"]})
    depths = ds_raw.coords["depth"]
    qubit_names = []
    survival_prob_rows = []
    fitted_curve_rows = []

    for qi, qubit in enumerate(qubits):
        qname = getattr(qubit, "name", f"q{qi}")
        qubit_names.append(qname)

        state_data = _get_qubit_state_data(ds_raw, qname)
        if state_data is None:
            survival_prob_rows.append(np.full(len(depths), np.nan, dtype=float))
            fitted_curve_rows.append(np.full(len(depths), np.nan, dtype=float))
            continue

        survival_prob = np.mean(state_data, axis=0)
        survival_prob_rows.append(np.asarray(survival_prob, dtype=float))

        fitted_curve = fit_results.get(qname, {}).get("fitted_curve")
        if fitted_curve is not None and len(fitted_curve) == len(depths):
            fitted_curve_rows.append(np.asarray(fitted_curve, dtype=float))
        else:
            fitted_curve_rows.append(np.full(len(depths), np.nan, dtype=float))

    ds_fit = ds_fit.assign_coords(qubit=("qubit", qubit_names)).assign(
        survival_probability=(["qubit", "depth"], np.stack(survival_prob_rows, axis=0)),
        state_fit=(["qubit", "depth"], np.stack(fitted_curve_rows, axis=0)),
        alpha=("qubit", [fit_results[q]["alpha"] for q in qubit_names]),
        A=("qubit", [fit_results[q]["A"] for q in qubit_names]),
        B=("qubit", [fit_results[q]["B"] for q in qubit_names]),
        error_per_clifford=("qubit", [fit_results[q]["error_per_clifford"] for q in qubit_names]),
        clifford_fidelity=("qubit", [fit_results[q]["clifford_fidelity"] for q in qubit_names]),
        avg_gates_per_clifford=("qubit", [fit_results[q]["avg_gates_per_clifford"] for q in qubit_names]),
        error_per_gate=("qubit", [fit_results[q]["error_per_gate"] for q in qubit_names]),
        native_gate_fidelity=("qubit", [fit_results[q]["native_gate_fidelity"] for q in qubit_names]),
        success=("qubit", [fit_results[q]["success"] for q in qubit_names]),
    )
    return ds_fit, fit_results


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
