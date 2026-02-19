from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """

    figs = []
    for qp in qubit_pairs:
        qc = qp.qubit_control
        qt = qp.qubit_target

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        fig.suptitle(f"Qc: {qc.name}, Qt: {qt.name}")

        # Panel 1
        plot_panel_confusion(ax, fits, qp.name, cmap="Blues", annotate=True, prob_fmt="{:.3f}")

        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_panel_confusion(
    ax: plt.Axes,
    ds_fit: xr.Dataset,
    qubit_pair: str,
    cmap: str = "Blues",
    annotate: bool = True,
    prob_fmt: str = "{:.2f}",
) -> None:
    """
    Ax[1]: confusion-matrix-like plot of probability/count of density matrix for the phase.
    Annotates each cell with "counts/total" and probability when annotate=True.
    """
    # prepared_state / measured_state (fallback name 'initial_state' if present)
    prep_name = "prepared_state" if "prepared_state" in ds_fit.prob.dims else "initial_state"
    meas_name = "measured_state"

    P = ds_fit.prob.sel(qubit_pair=qubit_pair)
    C = ds_fit.counts.sel(qubit_pair=qubit_pair)
    T = ds_fit.total_counts.sel(qubit_pair=qubit_pair)

    row_labels = list(P.coords[prep_name].values)
    col_labels = list(P.coords[meas_name].values)

    im = ax.imshow(P.values, vmin=0.0, vmax=1.0, cmap=cmap, origin="upper")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Measured state")
    ax.set_ylabel("Prepared state")
    ax.set_title(f"{qubit_pair}")

    # grid
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=1.0, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("auto")  # <-- same height as others

    if annotate:
        Pnp = P.values
        Cnp = C.values
        for i in range(Pnp.shape[0]):
            tot_i = int(T.isel({prep_name: i}).values)
            for j in range(Pnp.shape[1]):
                cnt = int(Cnp[i, j])
                prob = float(Pnp[i, j])
                text = f"{cnt}/{tot_i}\n{prob_fmt.format(prob)}"
                color = "black" if prob < 0.6 else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

