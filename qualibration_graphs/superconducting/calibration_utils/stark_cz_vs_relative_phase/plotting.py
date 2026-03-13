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

    for qp in qubit_pairs:
        qc, qt = qp.qubit_control, qp.qubit_target

        # Slice once per pair
        ds_pair = ds.sel(qubit_pair=qp.name)
        fits_pair = fits.sel(qubit_pair=qp.name)
        relative_phases = ds_pair.relative_phase.data
        control_targets = ds_pair.control_target.data
        control_states = ds_pair.control_state.data

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        for col, ct in enumerate(control_targets):
            ax = axes[col]

            for st in control_states:
                y = ds_pair.sel(control_target=ct, control_state=st)[val].data
                ax.plot(relative_phases, y, label=f"q{ct}=|{st}>", marker="o", linestyle="-")
            ax.set_ylim([-0.05, 1.05])
            ax.vlines(x=fits_pair.best_relative_phase.item(), ymin=-0.05, ymax=1.05, color="g", alpha=0.5)

            # y-labels on left
            ax.set_ylabel("<Z>") if col == 0 else None
            # ax.set_yticklabels([]) if col > 0 else None
            # titles (top row)
            ax.set_title(f"Q{ct}: {qc.name}") if ct == "c" else None
            ax.set_title(f"Q{ct}: {qt.name}") if ct == "t" else None
            # Bottom x-labels
            ax.set_xlabel(f"Relative Phase [2pi]")
            ax.legend()

        # fig.tight_layout()
        figs.append(fig)

    return figs
