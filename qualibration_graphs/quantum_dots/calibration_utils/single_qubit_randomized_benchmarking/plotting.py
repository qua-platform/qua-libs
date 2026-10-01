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
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import (
    apply_qubit_outcome_style,
    empty_figure,
    qubit_success,
)


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


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    ds_fit: xr.Dataset | None = None,
    fit_results: dict[str, dict[str, Any]] | None = None,
) -> Figure:
    """Create a multi-panel RB figure (one row per qubit).

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``depth`` and ``circuit`` coordinates and a
        stacked ``state(qubit, circuit, depth)`` array or legacy
        ``state_<qubit>`` variables.
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
    if n_qubits == 0:
        return empty_figure("No qubits selected for single-qubit RB.")

    fig, axes = plt.subplots(
        n_qubits,
        1,
        figsize=(8, 4.0 * n_qubits),
        squeeze=False,
    )

    depths = ds_raw.coords["depth"].values.astype(np.float64)

    for idx, qubit in enumerate(qubits):
        ax = axes[idx, 0]
        qname = getattr(qubit, "name", f"q{idx}")
        fit_results = fit_results or {}
        success = qubit_success(fit_results, qname)

        state_data = _get_qubit_state_data(ds_raw, qname)
        if state_data is None:
            apply_qubit_outcome_style(ax, qname, success, subtitle="No data")
            continue

        if ds_fit is not None and "survival_probability" in ds_fit.data_vars:
            survival_prob = ds_fit.survival_probability.sel(qubit=qname, drop=True).transpose("depth").values.astype(float)
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
        if ds_fit is not None and "state_fit" in ds_fit.data_vars:
            fitted = ds_fit.state_fit.sel(qubit=qname, drop=True).transpose("depth").values.astype(float)
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

        ax.text(
            0.95,
            0.95,
            (f"Native fidelity: {fidelity * 100:.2f}%\n" f"Error/Clifford: {epc * 100:.3f}%\n" f"α = {alpha_val:.5f}"),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        apply_qubit_outcome_style(ax, qname, success, subtitle="RB decay")
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
