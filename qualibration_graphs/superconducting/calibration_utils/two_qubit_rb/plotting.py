"""Plotting utilities for two-qubit randomized benchmarking experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from calibration_utils.two_qubit_rb.analysis import rb_decay_curve


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    *,
    interleaved: bool = False,
    title_prefix: str = "2Q Randomized Benchmarking",
    use_input_stream: bool | None = None,
) -> Dict[str, Figure]:
    """Plot RB survival curves on a chip-topology grid, one panel per qubit pair."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_data_with_fit(ax, ds_fit, qp_name, interleaved=interleaved)

    if use_input_stream is not None:
        stream_label = "(with input stream)" if use_input_stream else "(without input stream)"
        title_prefix = f"{title_prefix}\n{stream_label}"

    grid.fig.suptitle(title_prefix)
    grid.fig.tight_layout()
    return {"rb": grid.fig}


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    *,
    interleaved: bool = False,
) -> None:
    """Plot one qubit-pair RB survival curve using only the fitted dataset."""
    if qp_name not in ds_fit.qubit_pair.values:
        ax.text(0.5, 0.5, "No RB data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Qubit pair: {qp_name}")
        return

    fr = ds_fit.sel(qubit_pair=qp_name)
    depths = np.asarray(fr.circuit_depth.values, dtype=float)
    survival = fr.survival_probability.values
    stderr = fr.survival_stderr.values

    ax.errorbar(
        depths,
        survival,
        yerr=stderr,
        fmt=".",
        capsize=2,
        elinewidth=0.5,
        color="blue",
        label="Experimental Data",
    )

    smooth_depths = np.linspace(depths[0], depths[-1], 100)
    ax.plot(
        smooth_depths,
        rb_decay_curve(
            smooth_depths,
            float(fr.fit_amplitude.values),
            float(fr.fit_alpha.values),
            float(fr.fit_offset.values),
        ),
        color="red",
        linestyle="--",
        label="Exponential Fit",
    )

    if "standard_rb_overlay_survival" in fr and np.isfinite(fr.standard_rb_overlay_survival.values).any():
        ax.plot(
            depths,
            fr.standard_rb_overlay_survival.values,
            "s",
            color="green",
            alpha=0.5,
            label="StandardRB Data",
        )
        ax.plot(
            depths,
            fr.standard_rb_overlay_fitted.values,
            color="green",
            linestyle="--",
            label=f"StandardRB Fit (alpha={float(fr.standard_rb_fit_alpha.values):.4f})",
        )

    fidelity = float(fr.fidelity.values) * 100
    success = bool(fr.success.values) if "success" in fr.coords else True
    if interleaved:
        stats = f"CZ Fidelity = {fidelity:.2f}%"
    else:
        avg_gate_fid = fr.average_gate_fidelity.values if "average_gate_fidelity" in fr else np.nan
        stats = (
            f"2Q Clifford Fidelity = {fidelity:.2f}%\n"
            f"Single 2Q Gate Fidelity = {100 * float(avg_gate_fid):.2f}%"
        )

    title = f"Qubit pair: {qp_name}" if success else f"Qubit pair: {qp_name} - fit failed"
    ax.set_title(f"{title}\n{stats}", fontsize=9, linespacing=1.3, pad=10)

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel(r"Probability to recover to $|00\rangle$")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
