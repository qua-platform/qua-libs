"""Plotting utilities for Ramsey-based qubit flux long distortion visualizations."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter

from calibration_utils.qubit_flux_long_distortion_qubitspec.plotting import plot_flux_response


def annotate_branch_risk(fig, ds: xr.Dataset) -> bool:
    """Stamp a warning on ``fig`` if any qubit has branch-aliasing risk metadata."""
    if "branch_risk_code" not in ds:
        return False
    qnames = [str(n) for n in ds["branch_risk_code"].coords["qubit"].values]
    lines = []
    for nm in qnames:
        code = int(ds["branch_risk_code"].sel(qubit=nm).values)
        if code < 1:
            continue
        sw = float(ds["branch_sig_swing"].sel(qubit=nm).values)
        rs = float(ds["branch_ref_span"].sel(qubit=nm).values)
        tag = "HIGH" if code >= 2 else "marginal"
        detail = f"{nm}: {tag} — phase swing {sw:.2f}×2π, ref span {rs:.2f}×2π"
        if "branch_out_of_range" in ds:
            oor = float(ds["branch_out_of_range"].sel(qubit=nm).values)
            if oor > 0:
                detail += f", {oor:.0%} of points dropped (outside ref window)"
        lines.append(detail)
    if not lines:
        return False
    msg = (
        "⚠ BRANCH-ALIASING RISK: per-point phase→flux inversion (np.round) may be unreliable\n"
        + "\n".join(lines)
        + "\n(true phase approaches/exceeds one 2π window; see _compute_flux_response)"
    )
    fig.text(
        0.5,
        0.01,
        msg,
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round", facecolor="crimson", alpha=0.9),
    )
    try:
        fig.subplots_adjust(bottom=0.22)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Per-axis plotters (qubit-agnostic; used by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def plot_signal_phase(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    *,
    log_scale: bool = False,
) -> None:
    """Plot extracted Ramsey signal phase vs delay on one axis."""
    qname = qubit["qubit"]
    times = ds.time.values
    phase = ds.sel(qubit=qname).signal_phase.values
    ax.plot(times, phase, marker="o", ms=4, lw=1.2)
    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, which="both")
    else:
        ax.grid(True)
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Ramsey phase (rad)", fontsize=14)
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)


def plot_ramsey_fringe(
    ax: Axes,
    ds_raw: xr.Dataset,
    qubit: dict,
    fig: plt.Figure,
    *,
    log_scale: bool = False,
) -> None:
    """Plot Ramsey signal vs (time, frame rotation) on one axis."""
    if "state" in ds_raw.data_vars:
        signal_var = "state"
        cbar_label = "State"
    elif "I" in ds_raw.data_vars:
        signal_var = "I"
        cbar_label = "I (V)"
    else:
        ax.set_title(f"{qubit['qubit']} — no signal")
        return

    qname = qubit["qubit"]
    q_ds = ds_raw.sel(qubit=qname)
    if "frame" not in q_ds[signal_var].dims:
        ax.set_title(f"{qname} — no frame axis")
        return

    times = np.asarray(q_ds.time.values, dtype=float)
    frames = np.asarray(q_ds.frame.values, dtype=float)
    signal = np.asarray(q_ds[signal_var].transpose("frame", "time").values, dtype=float)
    im = ax.pcolormesh(times, frames, signal, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, which="both")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Frame rotation (×2π)")
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)


def plot_ref_phase_cal(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
) -> None:
    """Plot reference Ramsey phase vs probe amplitude on one axis."""
    qname = qubit["qubit"]
    if "ref_phase_cal" not in ds:
        ax.set_title(f"{qname} — no reference calibration")
        return
    ds["ref_phase_cal"].sel(qubit=qname).plot(ax=ax, marker=".")
    ax.set_xlabel("Ramsey flux amp (V)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title(qname)
    ax.grid(True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict,
    *,
    debug: bool = False,
    ds_raw: Optional[xr.Dataset] = None,
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default figures: flux response (with IIR fit) only.

    With ``debug=True``: Ramsey fringe heatmap (time × frame), extracted Ramsey
    phase vs time, and the reference phase-vs-amplitude calibration curve.
    """
    grid_locations = [q.grid_location for q in qubits]
    figures: Dict[str, plt.Figure] = {}

    grid_flux = QubitGrid(ds, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        plot_flux_response(
            ax,
            ds,
            qubit,
            fit=fit_results.get(qubit["qubit"]),
            log_scale=log_scale,
        )
    grid_flux.fig.suptitle("Flux response vs time after flux pulse", fontsize=16)
    grid_flux.fig.set_size_inches(15, 9)
    grid_flux.fig.tight_layout()
    annotate_branch_risk(grid_flux.fig, ds)
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    if ds_raw is not None:
        signal_key = "state" if "state" in ds_raw.data_vars else "I"
        if signal_key in ds_raw.data_vars and "frame" in ds_raw[signal_key].dims:
            grid_fringe = QubitGrid(ds_raw, grid_locations)
            for ax, qubit in grid_iter(grid_fringe):
                plot_ramsey_fringe(ax, ds_raw, qubit, grid_fringe.fig, log_scale=log_scale)
            grid_fringe.fig.suptitle("Debug: Ramsey signal vs (time, frame rotation)", fontsize=16)
            grid_fringe.fig.set_size_inches(15, 9)
            grid_fringe.fig.tight_layout()
            figures["ramsey_fringe"] = grid_fringe.fig

    if "signal_phase" in ds:
        grid_phase = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_phase):
            plot_signal_phase(ax, ds, qubit, log_scale=log_scale)
        grid_phase.fig.suptitle("Debug: Ramsey phase vs time after flux pulse", fontsize=16)
        grid_phase.fig.set_size_inches(15, 9)
        grid_phase.fig.tight_layout()
        figures["signal_phase"] = grid_phase.fig

    if "ref_phase_cal" in ds:
        grid_ref = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_ref):
            plot_ref_phase_cal(ax, ds, qubit)
        grid_ref.fig.suptitle("Debug: Reference Ramsey phase vs flux amplitude", fontsize=16)
        grid_ref.fig.set_size_inches(15, 9)
        grid_ref.fig.tight_layout()
        figures["ref_phase_cal"] = grid_ref.fig

    return figures
