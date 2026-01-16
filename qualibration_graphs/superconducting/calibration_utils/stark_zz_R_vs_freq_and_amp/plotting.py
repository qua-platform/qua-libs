from typing import List, Dict, Optional
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter

from qualibration_libs.analysis import oscillation_decay_exp
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    fits: xr.Dataset,
) -> List[Figure]:
    """
    Plot raw Ramsey/echo-like traces for Control (left col) and Target (right col),
    and overlay the fitted curve ONLY on the Target axes.

    For each qubit-pair:
      rows -> control_state in {0,1}
      cols -> Control (C), Target (T)

    The observable is 'state' if present, otherwise 'I'.
    """
    figs: List[Figure] = []
    val = "state" if "state" in ds.data_vars else "I"
    
    cmap = plt.get_cmap("viridis")          # or "cividis", "plasma"
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # Slice once per pair
        ds_pair = ds.sel(qubit_pair=qp.name)
        amp_scalings = ds_pair.amp_scaling.data  # time in ns
        detunings = ds_pair.detuning.data  # time in ns

        fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True)
        basis_map = {0: "x", 1: "y", 2: "z"}

        for row, bs in enumerate([0, 1, 2]):
            for col, ct_st in enumerate([["c", 0], ["c", 1], ["t", 0], ["t", 1]]):
                ax = axes[row, col]
                ct, st = ct_st

                # raw data
                y = ds_pair.sel(control_target=ct, target_basis=bs, control_state=st)[val].data

                # Use a shared norm/cmap so all panels use the same color scale
                ax.pcolor(
                    amp_scalings, detunings / 1e6, y,
                    cmap=cmap, norm=norm, shading="auto"
                )

                # y-labels on left
                ax.set_ylabel(f"<{basis_map[bs]}>\nFrequency Detuning [MHz]") if col == 0 else None
                ax.set_yticklabels([]) if col > 0 else None
                # titles (top row)
                ax.set_title(f"Q{ct}: {qc.name} (qc=|{st}>)") if row == 0 and ct == "c" else None
                ax.set_title(f"Q{ct}: {qt.name} (qc=|{st}>)") if row == 0 and ct == "t" else None
                # Bottom x-labels
                ax.set_xlabel("Amplitude Scaling") if row == 2 else None

        # One shared colorbar across all subplots, consistent with [0, 1]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Matplotlib quirk for colorbar without a specific mappable
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.9, pad=0.025, location="right")
        cbar.set_label(val)

        # fig.tight_layout()
        figs.append(fig)

    return figs


def plot_fit_summary(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    ds_fit: xr.Dataset,
) -> List[plt.Figure]:
    """
    For each qubit pair, create a 1x2 figure:
      (0) relative phase vs freq offset for control_state = 0 and 1
      (1) relative phase vs zz_coeff

    Uses variables prepared in ds_fit:
      - freq_offset(qubit_pair, control_target, control_state, relative_phase) [MHz]
      - zz_coeff(qubit_pair, relative_phase) [MHz if you convert, else same as freq_offset units]
      - best_phase(qubit_pair)
      - best_zz_coeff(qubit_pair)
    """
    figs: List[plt.Figure] = []

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # --- Pull data for this pair
        R = ds_fit.R.sel(qubit_pair=qp.name)
        amp_scalings = R.amp_scaling.data
        detunings = R.detuning.data

        # --- Value at that location
        best_detuning = ds_fit.sel(qubit_pair=qp.name).best_detuning.item()
        best_amp_scaling = ds_fit.sel(qubit_pair=qp.name).best_amp_scaling.item()
        best_value = ds_fit.sel(qubit_pair=qp.name).best_R.item()

        # --- Make figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name}")

        pcm = ax.pcolor(amp_scalings, detunings / 1e6, R.values, shading="auto")

        ### Add star annotation
        ax.plot(
            best_amp_scaling, best_detuning / 1e6, marker="*", markersize=15,
            markeredgecolor=(0, 0, 0, 0.3), markerfacecolor="yellow", zorder=5,
        )

        # Smart-ish offset so the label doesn't run off the edge
        dx = -14 if best_amp_scaling > np.mean(amp_scalings) else 8
        dy = -10 if (best_detuning / 1e6) > (detunings / 1e6).mean() else 8

        # Text annotation with a light box for readability
        ax.annotate(
            f"{best_value:.4f}",
            xy=(best_amp_scaling, best_detuning / 1e6),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            color=(0.2, 0.2, 0.2),
            # weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", lw=0, alpha=0.2),  # no edge color
            zorder=6,
        )

        ax.set_aspect("auto")
        ax.set_xlabel("Amplitude Scaling")      ### (x-axis = amp_scaling)
        ax.set_ylabel("Frequency Detuning [MHz]")

        fig.colorbar(pcm, ax=ax, label="R value")   ### optional colorbar

        fig.tight_layout()
        figs.append(fig)

    return figs