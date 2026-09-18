"""Plotting utilities for residual iSWAP error amplification."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _plot_component(
    ax: plt.Axes,
    n_cycles: np.ndarray,
    ds_fit: xr.Dataset,
    pair_name: str,
    axis_name: str,
    *,
    theta: float,
) -> None:
    transfer_key = f"transfer_{axis_name}_{pair_name}"
    fit_key = f"transfer_{axis_name}_fit_{pair_name}"
    transfer_10_key = f"transfer_10_{axis_name}_{pair_name}"
    transfer_01_key = f"transfer_01_{axis_name}_{pair_name}"

    if transfer_key not in ds_fit.data_vars:
        ax.set_title(f"{pair_name} - no {axis_name} data")
        return

    ax.plot(
        n_cycles,
        ds_fit[transfer_10_key].values,
        "o",
        ms=3,
        alpha=0.45,
        label="|10> transfer",
    )
    ax.plot(
        n_cycles,
        ds_fit[transfer_01_key].values,
        "s",
        ms=3,
        alpha=0.45,
        label="|01> transfer",
    )
    ax.plot(
        n_cycles,
        ds_fit[transfer_key].values,
        "o",
        color="k",
        ms=4,
        alpha=0.75,
        label="mean",
    )
    if fit_key in ds_fit.data_vars:
        ax.plot(
            n_cycles,
            ds_fit[fit_key].values,
            "-",
            color="C3",
            lw=1.5,
            label="fit",
        )

    dd_label = "X⊗X" if axis_name == "x" else "Y⊗X"
    ax.set_title(f"{dd_label}: |theta|={theta:.4g} rad")
    ax.set_xlabel("Two-cycle repetitions")
    ax.set_ylabel("Odd-subspace transfer")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,  # noqa: ARG001
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",  # noqa: ARG001
) -> plt.Figure:
    """Create the residual iSWAP error-amplification summary figure."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs,
        2,
        figsize=(12, max(3.8, 3.4 * n_pairs)),
        squeeze=False,
    )

    if ds_fit is None:
        for row in axes:
            for ax in row:
                ax.set_title("No fit data")
        return fig

    n_cycles = ds_fit.coords["num_cphase_cycles"].values.astype(np.float64)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        result = fit_results.get(name, {})
        _plot_component(
            axes[pair_idx, 0],
            n_cycles,
            ds_fit,
            name,
            "x",
            theta=float(result.get("theta_x", np.nan)),
        )
        _plot_component(
            axes[pair_idx, 1],
            n_cycles,
            ds_fit,
            name,
            "y",
            theta=float(result.get("theta_y", np.nan)),
        )
        status = "OK" if result.get("success") else "FAILED"
        theta_abs = float(result.get("theta_iswap_abs", np.nan))
        axes[pair_idx, 0].text(
            0.02,
            0.96,
            f"{name} [{status}] |theta|={theta_abs:.4g} rad",
            transform=axes[pair_idx, 0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )

    fig.suptitle("Residual iSWAP Error Amplification", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
