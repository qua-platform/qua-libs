"""Plotting utilities for single-qubit randomized benchmarking.

Generates a multi-row figure (one row per qubit) showing:
  - Survival probability vs circuit depth (scatter + error bars).
  - Fitted exponential decay  F(m) = A · α^m + B.
  - Annotated Clifford fidelity, error per Clifford, and α.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    ds_fit: xr.Dataset | None = None,
    fit_results: dict[str, dict[str, Any]] | None = None,
) -> plt.Figure:
    """Create a multi-panel RB figure (one row per qubit).

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``depth`` and ``circuit`` coordinates and
        ``state_<qubit>`` variables shaped ``[num_circuits, num_depths]``.
    ds_fit : xr.Dataset or None
        Optional fit dataset containing survival probabilities and fitted
        curves vs depth.
    fit_results : dict or None
        Output of :func:`~.analysis.fit_raw_data`.
    qubits : list
        Qubit objects (each must have a ``.name`` attribute).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits,
        1,
        figsize=(8, 4.0 * n_qubits),
        squeeze=False,
    )

    depths = ds_raw.coords["depth"].values.astype(np.float64)

    for idx, qubit in enumerate(qubits):
        ax = axes[idx, 0]
        qname = qubit.name
        fit_results = fit_results or {}

        state_data = _get_qubit_state_data(ds_raw, qname)
        if state_data is None:
            ax.set_title(f"{qname} — no data")
            continue

        if ds_fit is not None and f"survival_probability_{qname}" in ds_fit.data_vars:
            survival_prob = ds_fit[f"survival_probability_{qname}"].values
        else:
            survival_prob = np.mean(state_data, axis=0)
        n_circuits = state_data.shape[0]

        # Binomial standard error
        std_err = np.sqrt(survival_prob * (1 - survival_prob) / max(n_circuits, 1))

        r = fit_results.get(qname, {})

        # Data points
        ax.errorbar(
            depths,
            survival_prob,
            yerr=std_err,
            fmt="o",
            ms=4,
            capsize=3,
            color="C0",
            label="data",
        )

        # Fitted curve
        fitted = None
        if ds_fit is not None and f"fitted_curve_{qname}" in ds_fit.data_vars:
            fitted = ds_fit[f"fitted_curve_{qname}"].values
        elif r.get("fitted_curve") is not None:
            fitted = r.get("fitted_curve")

        if fitted is not None and len(fitted) == len(depths):
            x_smooth = np.linspace(float(depths.min()), float(depths.max()), 200)
            alpha = r.get("alpha", 0)
            A = r.get("A", 0)
            B = r.get("B", 0)
            y_smooth = A * alpha**x_smooth + B
            ax.plot(x_smooth, y_smooth, "-", lw=2, color="C1", label="fit")

        # Annotation
        fidelity = r.get("native_gate_fidelity", float("nan"))
        epc = r.get("error_per_clifford", float("nan"))
        alpha_val = r.get("alpha", float("nan"))
        status = "OK" if r.get("success") else "FAIL"

        ax.text(
            0.95,
            0.95,
            (
                f"Native fidelity: {fidelity * 100:.2f}%\n"
                f"Error/Clifford: {epc * 100:.3f}%\n"
                f"α = {alpha_val:.5f}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        ax.set_title(f"{qname}  [{status}]")
        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel("Survival probability")
        # ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Single-Qubit Randomized Benchmarking", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_all(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    *,
    ds_fit: xr.Dataset | None = None,
    fit_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, plt.Figure]:
    """Build and return all RB figures."""
    return {
        "raw_data_with_fit": plot_raw_data_with_fit(
            ds_raw,
            qubits,
            ds_fit=ds_fit,
            fit_results=fit_results,
        )
    }
