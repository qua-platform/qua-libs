"""Plotting utilities for the measurement-induced dephasing matrix experiment.

The three figures reproduce Fig. 6 of Phys. Rev. Applied 23, 054089 (arXiv:2412.14853):
  * :func:`plot_contrast_with_fit` is panel (a), the relative echo contrast versus the relative
    readout amplitude;
  * :func:`plot_dephasing_matrix` is panel (b), the dephasing-rate colour map;
  * :func:`plot_phase_oscillations` is the underlying raw data, used for debugging.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _decay_model(xi: np.ndarray, c0: float, gamma: float, tau_p: float) -> np.ndarray:
    """The fitted contrast decay c(xi) = c0 * exp(-Gamma * tau_p * xi**2)."""
    return c0 * np.exp(-gamma * tau_p * xi**2)


def _plot_pair(ax, ds_fit: xr.Dataset, qubit_name: str, driven: str, label=None, colour=None, marker_size=4):
    """Plot the relative contrast of one (measured qubit, driven resonator) pair with its fit.

    Returns the colour used, so that the same qubit keeps the same colour in the inset.
    """
    pair = ds_fit.sel(qubit=qubit_name, driven_resonator=driven)
    xi = pair.xi.values
    line = ax.plot(xi, pair.contrast_relative.values, "o", ms=marker_size, label=label, color=colour)[0]

    gamma = float(pair.Gamma)
    if np.isfinite(gamma):
        xi_dense = np.linspace(0, xi.max(), 201)
        ax.plot(
            xi_dense,
            _decay_model(xi_dense, 1.0, gamma, float(pair.tau_p)),
            "--",
            color=line.get_color(),
            lw=1,
        )
    return line.get_color()


def plot_contrast_with_fit(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot the relative echo contrast c/c0 against the relative readout amplitude xi.

    One panel per driven resonator, one curve per measured qubit, with the fitted exponential
    overlaid. The vertical axis is logarithmic because the diagonal element decays by orders of
    magnitude over the swept range.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.
    qubits : list of AnyTransmon
        The measured qubits, used only for their names.

    Returns
    -------
    Figure
        The matplotlib figure containing the plots.
    """
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    num_panels = len(driven_names)
    num_columns = min(num_panels, 2)
    num_rows = int(np.ceil(num_panels / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(7 * num_columns, 4.5 * num_rows), squeeze=False)

    for panel_index, driven in enumerate(driven_names):
        ax = axes.flat[panel_index]
        colours = {}
        for qubit in qubits:
            colours[qubit.name] = _plot_pair(ax, ds_fit, qubit.name, driven, label=qubit.name)
        ax.set_yscale("log")
        ax.set_xlabel("Relative readout amplitude $\\xi$")
        ax.set_ylabel("Relative echo contrast $c/c_0$")
        ax.set_title(f"Driving {driven}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # The diagonal element decays by orders of magnitude while the crosstalk barely moves, so on
        # the shared logarithmic axis the off-diagonal curves collapse onto c/c0 = 1. They are
        # repeated in an inset on a linear, auto-scaled axis, as in the inset of Fig. 6(a).
        off_diagonal_qubits = [qubit for qubit in qubits if qubit.name != driven]
        if not off_diagonal_qubits:
            continue
        inset = ax.inset_axes([0.45, 0.12, 0.5, 0.42])
        for qubit in off_diagonal_qubits:
            _plot_pair(inset, ds_fit, qubit.name, driven, colour=colours[qubit.name], marker_size=3)
        inset.set_title("off-diagonal only", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.3)

    for empty_index in range(num_panels, axes.size):
        axes.flat[empty_index].axis("off")

    fig.suptitle("Measurement-induced dephasing: echo contrast vs readout amplitude")
    fig.tight_layout()
    return fig


def _readable_text_colour(image, value: float) -> str:
    """Return black or white, whichever stays legible on the cell colour holding ``value``.

    Both colour maps run from near-white at their low end to dark at their high end, so a single
    fixed text colour is unreadable at one end of the scale. The choice is made on the perceived
    luminance of the cell.
    """
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "k" if luminance > 0.55 else "w"


def plot_dephasing_matrix(ds_fit: xr.Dataset) -> Figure:
    """Plot the measurement-induced dephasing rates as a single matrix with two colour bars.

    The self-dephasing (diagonal) rates and the crosstalk (off-diagonal) rates differ by several
    orders of magnitude, so a single colour scale would saturate. Following Fig. 6(b) of
    Phys. Rev. Applied 23, 054089, both are drawn on one set of axes as two superimposed layers,
    each masked to its own cells and carrying its own colour bar: the off-diagonal in Hz and the
    diagonal in MHz. The colour maps are the ones used in that figure, 'Reds' for the crosstalk and
    'Greys' for the self-dephasing.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.

    Returns
    -------
    Figure
        The matplotlib figure containing the dephasing matrix.
    """
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    gamma = ds_fit.Gamma.transpose("qubit", "driven_resonator").values
    is_diagonal = ds_fit.is_diagonal.transpose("qubit", "driven_resonator").values

    off_diagonal_only = np.ma.masked_invalid(np.where(is_diagonal, np.nan, gamma))
    diagonal_only = np.ma.masked_invalid(np.where(is_diagonal, gamma * 1e-6, np.nan))

    fig, ax = plt.subplots(figsize=(1.1 * len(driven_names) + 6, 1.1 * len(qubit_names) + 2.5))

    # The colour bar axes are placed explicitly rather than stacked automatically, so that the
    # off-diagonal scale - the actual crosstalk result - sits closest to the matrix and each bar has
    # room for its own label. The pad of the outer bar must clear the label of the inner one.
    divider = make_axes_locatable(ax)
    layers = {}
    for values, cmap_name, label, pad in (
        (off_diagonal_only, "Reds", "Off-diagonal dephasing rate (Hz)", 0.15),
        (diagonal_only, "Greys", "Diagonal dephasing rate (MHz)", 0.85),
    ):
        # Masked cells are drawn fully transparent so the two layers do not hide one another.
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)
        image = ax.imshow(values, cmap=cmap, origin="upper", interpolation="nearest")
        colour_axes = divider.append_axes("right", size="4%", pad=pad)
        fig.colorbar(image, cax=colour_axes, label=label)
        layers[label] = image

    ax.set_xticks(range(len(driven_names)), driven_names)
    ax.set_yticks(range(len(qubit_names)), qubit_names)
    ax.set_xlabel("Resonator driven")
    ax.set_ylabel("Qubit measured")
    # Grid lines on the cell boundaries, to separate the two colour scales visually. A neutral grey
    # is used because both colour maps are near-white at their low end.
    ax.set_xticks(np.arange(len(driven_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(qubit_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.6", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    off_diagonal_image = layers["Off-diagonal dephasing rate (Hz)"]
    diagonal_image = layers["Diagonal dephasing rate (MHz)"]
    for row in range(len(qubit_names)):
        for column in range(len(driven_names)):
            on_diagonal = is_diagonal[row, column]
            values = diagonal_only if on_diagonal else off_diagonal_only
            image = diagonal_image if on_diagonal else off_diagonal_image
            value = values[row, column]
            if value is np.ma.masked or not np.isfinite(value):
                continue
            ax.text(
                column,
                row,
                f"{value:.3g}",
                ha="center",
                va="center",
                color=_readable_text_colour(image, value),
                fontsize=8,
            )

    ax.set_title("Measurement-induced dephasing matrix")
    fig.tight_layout()
    return fig


def plot_phase_oscillations(ds_fit: xr.Dataset, node) -> Figure:
    """Plot every phase oscillation together with its fitted sinusoid.

    This is the raw data behind the contrast extraction: one panel per (measured qubit, driven
    resonator) pair, each holding the oscillations for all swept amplitudes. It is the figure to
    look at when a dephasing-rate fit misbehaves.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.
    node : QualibrationNode
        Node whose parameters determine which signal was measured.

    Returns
    -------
    Figure
        The matplotlib figure containing the grid of oscillations.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    phases = ds_fit.phase.values
    phases_dense = np.linspace(phases.min(), phases.max(), 201)

    fig, axes = plt.subplots(
        len(qubit_names),
        len(driven_names),
        figsize=(3.2 * len(driven_names), 2.6 * len(qubit_names)),
        squeeze=False,
        sharex=True,
    )
    colours = plt.cm.viridis(np.linspace(0, 1, ds_fit.sizes["xi_idx"]))

    for row, qubit_name in enumerate(qubit_names):
        for column, driven in enumerate(driven_names):
            ax = axes[row][column]
            pair = ds_fit.sel(qubit=qubit_name, driven_resonator=driven)
            for xi_index in range(ds_fit.sizes["xi_idx"]):
                point = pair.isel(xi_idx=xi_index)
                ax.plot(phases, point[signal_name].values, ".", ms=3, color=colours[xi_index])
                fitted = (
                    float(point.oscillation_offset)
                    + float(point.contrast) * np.cos(phases_dense - float(point.oscillation_phase))
                )
                ax.plot(phases_dense, fitted, "-", lw=0.8, color=colours[xi_index])
            ax.set_title(f"{qubit_name} | {driven}", fontsize=9)
            if row == len(qubit_names) - 1:
                ax.set_xlabel("Final pulse phase [rad]")
            if column == 0:
                ax.set_ylabel(signal_name)

    fig.suptitle("Echo phase oscillations (colour: relative readout amplitude)")
    fig.tight_layout()
    return fig
