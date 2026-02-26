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


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    fit_results: dict[str, dict[str, Any]],
    qubits: list[Any],
) -> plt.Figure:
    """Create a multi-panel RB figure (one row per qubit).

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``depth`` and ``circuit`` coordinates and
        ``state_<qubit>`` variables shaped ``[num_circuits, num_depths]``.
    fit_results : dict
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
        var_name = f"state_{qname}"

        if var_name not in ds_raw.data_vars:
            ax.set_title(f"{qname} — no data")
            continue

        state_data = ds_raw[var_name].values  # [num_circuits, num_depths]
        survival_prob = np.mean(state_data, axis=0)
        n_circuits = state_data.shape[0]

        # Binomial standard error
        std_err = np.sqrt(
            survival_prob * (1 - survival_prob) / max(n_circuits, 1)
        )

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
        fitted = r.get("fitted_curve")
        if fitted is not None and len(fitted) == len(depths):
            x_smooth = np.linspace(float(depths.min()), float(depths.max()), 200)
            alpha = r.get("alpha", 0)
            A = r.get("A", 0)
            B = r.get("B", 0)
            y_smooth = A * alpha ** x_smooth + B
            ax.plot(x_smooth, y_smooth, "-", lw=2, color="C1", label="fit")

        # Annotation
        fidelity = r.get("gate_fidelity", float("nan"))
        epc = r.get("error_per_clifford", float("nan"))
        alpha_val = r.get("alpha", float("nan"))
        status = "OK" if r.get("success") else "FAIL"

        ax.text(
            0.95,
            0.95,
            (
                f"Clifford fidelity: {fidelity * 100:.2f}%\n"
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
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Single-Qubit Randomized Benchmarking", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return fig
