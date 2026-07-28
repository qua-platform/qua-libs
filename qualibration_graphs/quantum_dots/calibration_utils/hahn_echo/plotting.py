"""Plotting utilities for the Hahn echo T₂ measurement."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.plot_style import (
    apply_node_outcome_style,
    node_success,
)
from calibration_utils.measurement_utils import get_parity_item_names

_SIGNAL_LABELS = {
    "E_p1_given_p0_0": "P(1 | empty dot)",
    "E_p1_given_p0_1": "P(1 | loaded dot)",
}


_TAU_XLABEL = "Idle delay τ [{unit}]"


def _tau_axis(tau_ns: np.ndarray) -> tuple[np.ndarray, str]:
    tau_plot = tau_ns.astype(float)
    if float(tau_plot.max()) > 5_000.0:
        return tau_plot / 1e3, "µs"
    return tau_plot, "ns"


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    *,
    analysis_signal: str = "E_p1_given_p0_0",
) -> dict[str, Figure]:
    """Create a multi-panel Hahn-echo figure (one column per qubit).

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset with data, fit curves, and summary scalars.
    qubits : list
        Qubit objects (names resolved from the dataset when possible).
    analysis_signal : str
        Which conditional expectation to plot.

    Returns
    -------
    dict[str, Figure]
        ``{"decay": figure}`` with all qubits as horizontal subplots.
    """
    qubit_names = get_parity_item_names(
        ds_fit,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return {"decay": fig}

    tau_ns = np.asarray(ds_fit.tau.values, dtype=float)
    x_plot, time_unit = _tau_axis(tau_ns)
    y_label = _SIGNAL_LABELS.get(analysis_signal, analysis_signal)

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax = axes[0, i]
        signal_var = f"{analysis_signal}_{qname}"
        fit_var = f"{analysis_signal}_fit_{qname}"
        success = node_success(ds_fit, qname)

        if signal_var in ds_fit:
            y = np.asarray(ds_fit[signal_var].values, dtype=float)
            ax.scatter(x_plot, y, c="C0", s=12, alpha=0.6, zorder=3, label="Data")
            ax.plot(x_plot, y, color="C0", lw=0.8, alpha=0.5)

        if success is not False and fit_var in ds_fit:
            fitted = np.asarray(ds_fit[fit_var].values, dtype=float)
            if np.any(np.isfinite(fitted)):
                ax.plot(
                    x_plot,
                    fitted,
                    "-",
                    lw=2,
                    color="C1",
                    alpha=0.9,
                    label="Exponential fit",
                )

        ax.set_xlabel(_TAU_XLABEL.format(unit=time_unit))
        ax.set_ylabel(y_label)
        ax.set_ylim(-0.05, 1.05)
        apply_node_outcome_style(ax, qname, success)

        if success is not False and "T2_echo" in ds_fit and qname in ds_fit.qubit.values:
            t2 = float(ds_fit["T2_echo"].sel(qubit=qname).values)
            amp = float(ds_fit["amplitude"].sel(qubit=qname).values)
            off = float(ds_fit["offset"].sel(qubit=qname).values)
            if time_unit == "µs" and np.isfinite(t2):
                t2_str = f"T₂_echo = {t2 / 1e3:.2f} µs"
            else:
                t2_str = f"T₂_echo = {t2:.1f} ns"
            ax.text(
                0.02,
                0.98,
                f"{t2_str}\nA = {amp:.4f},  offset = {off:.4f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
            ax.legend(loc="lower right", fontsize=8)
        elif success is not False:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Hahn echo T₂", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return {"decay": fig}
