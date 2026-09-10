"""Plotting helpers for readout confusion matrix calibrations."""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def get_state_labels(num_qubits: int) -> list[str]:
    """Return binary labels for all computational basis states."""
    return [format(state, f"0{num_qubits}b") for state in range(2**num_qubits)]


def axis_label_style(num_qubits: int) -> tuple[int, int, int, bool]:
    """Return tick rotation (deg), fontsize, step, and whether to use integer ticks."""
    if num_qubits <= 2:
        return 0, 10, 1, False
    if num_qubits == 3:
        return 45, 8, 1, False
    if num_qubits == 4:
        return 90, 7, 2, False
    return 0, 8, 4, True


def annotation_style(num_qubits: int) -> tuple[bool, int]:
    """Return whether to annotate cells and the font size to use."""
    if num_qubits <= 2:
        return True, 10
    if num_qubits == 3:
        return True, 8
    if num_qubits == 4:
        return False, 6
    return False, 4


def reset_type_subtitle(node) -> str:
    """Return a reset-type subtitle line when available on the node."""
    reset_type = getattr(getattr(node, "parameters", None), "reset_type", None)
    if reset_type is None:
        return ""
    return f"\nreset type = {reset_type}"


def diff_confusion_matrices(
    confusions: Dict[str, np.ndarray],
    kron_confs: Dict[str, np.ndarray],
    names: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Return direct-minus-Kron matrices for targets present in both inputs."""
    return {name: confusions[name] - kron_confs[name] for name in names if name in confusions and name in kron_confs}


def _plot_dual_colormap(
    ax: Axes,
    matrix: np.ndarray,
    num_states: int,
    *,
    colorbar_axes: Optional[Tuple[Axes, Axes]],
) -> Tuple[float, float]:
    """Draw ``matrix`` with two colour scales split at the diagonal.

    The diagonal (correct assignments, typically > 90%) and the off-diagonal (errors, typically a
    few %) live orders of magnitude apart, so a single scale renders one of them flat. The diagonal
    is drawn in greys and the off-diagonal in reds, each with its own colour bar (CS_installations
    convention).

    ``colorbar_axes``, when given, are a pre-built ``(diagonal_cax, off_diagonal_cax)`` pair --
    typically carved out of the panel's own grid cell by the caller (see
    ``plot_confusion_matrices_grid``) so both bars come out the same size regardless of matplotlib's
    automatic colorbar sizing, which does not guarantee that for two bars sharing one axes.

    Returns the ``(diagonal_vmin, off_diagonal_vmax)`` scale bounds, used to pick annotation text
    colour.
    """
    diagonal_mask = np.eye(num_states, dtype=bool)
    diagonal = np.where(diagonal_mask, matrix, np.nan)
    off_diagonal = np.where(diagonal_mask, np.nan, matrix)

    diagonal_vmin = min(np.floor(100 * np.nanmin(diagonal)) / 100, 1.0 - 1e-3)
    off_diagonal_vmax = max(np.ceil(100 * np.nanmax(off_diagonal)) / 100, 1e-3)

    diagonal_cmap = plt.get_cmap("Greys").copy()
    off_diagonal_cmap = plt.get_cmap("Reds").copy()
    for colormap in (diagonal_cmap, off_diagonal_cmap):
        # The masked half of each image must let the other one show through.
        colormap.set_bad(alpha=0.0)

    image_off = ax.imshow(off_diagonal, vmin=0.0, vmax=off_diagonal_vmax, cmap=off_diagonal_cmap, origin="upper")
    image_diagonal = ax.imshow(diagonal, vmin=diagonal_vmin, vmax=1.0, cmap=diagonal_cmap, origin="upper")

    if colorbar_axes is not None:
        cax_diagonal, cax_off = colorbar_axes

        colorbar_diagonal = ax.figure.colorbar(image_diagonal, cax=cax_diagonal)
        colorbar_diagonal.formatter = PercentFormatter(xmax=1, decimals=0)
        colorbar_diagonal.ax.tick_params(labelsize=8)
        colorbar_diagonal.set_label("diagonal (%)", fontsize=8)

        colorbar_off = ax.figure.colorbar(image_off, cax=cax_off)
        colorbar_off.formatter = PercentFormatter(xmax=1, decimals=1)
        colorbar_off.ax.tick_params(labelsize=8)
        colorbar_off.set_label("off-diagonal (%)", fontsize=8)

    return diagonal_vmin, off_diagonal_vmax


def plot_confusion_matrix_on_axes(
    ax: Axes,
    conf: Optional[np.ndarray],
    state_labels: Sequence[str],
    title: str,
    *,
    is_difference: bool = False,
    annotate_cells: bool = True,
    text_fontsize: int = 10,
    show_colorbar: bool = False,
    cmap: Optional[str] = None,
    colorbar_axes: Optional[Tuple[Axes, Axes]] = None,
) -> None:
    """Plot one confusion matrix stored as ``conf[measured, prepared]``.

    Non-difference matrices are drawn with two colour scales split at the diagonal (see
    ``_plot_dual_colormap``); the difference matrix keeps a single diverging colour scale.
    """
    if conf is None:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    matrix = np.asarray(conf)
    num_states = len(state_labels)
    ticks = np.arange(num_states)
    label_rotation, label_fontsize, label_step, use_integer_ticks = axis_label_style(int(np.log2(num_states)))
    tick_positions = ticks[::label_step]
    if use_integer_ticks:
        tick_labels = [str(i) for i in tick_positions]
    else:
        tick_labels = [state_labels[i] for i in tick_positions]

    if is_difference:
        max_abs = max(np.max(np.abs(matrix)), 1e-3)
        mesh = ax.imshow(matrix, cmap=cmap or "RdBu", vmin=-max_abs, vmax=max_abs, origin="upper")
        if show_colorbar:
            colorbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
            colorbar.ax.tick_params(labelsize=8)

        def cell_color(meas: int, prep: int, val: float) -> str:
            return "k" if abs(val) < 0.5 * max_abs else "w"

    else:
        diagonal_vmin, off_diagonal_vmax = _plot_dual_colormap(ax, matrix, num_states, colorbar_axes=colorbar_axes)

        def cell_color(meas: int, prep: int, val: float) -> str:
            if meas == prep:
                shade = (val - diagonal_vmin) / max(1.0 - diagonal_vmin, 1e-12)
            else:
                shade = val / off_diagonal_vmax
            return "w" if shade > 0.6 else "k"

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=label_rotation,
        fontsize=label_fontsize,
        ha="right" if label_rotation else "center",
    )
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=label_fontsize)

    if annotate_cells:
        for meas in range(num_states):
            for prep in range(num_states):
                val = matrix[meas, prep]
                if is_difference and abs(val) <= 0.01:
                    continue
                ax.text(
                    prep,
                    meas,
                    f"{100 * val:.1f}%",
                    ha="center",
                    va="center",
                    color=cell_color(meas, prep, val),
                    fontsize=text_fontsize,
                )

    ax.set_ylabel("measured" + (" state index" if use_integer_ticks else ""))
    ax.set_xlabel("prepared" + (" state index" if use_integer_ticks else ""))
    ax.set_title(title, fontsize=9)


def plot_marginal_confusion_matrix_on_axes(
    ax: Axes,
    conf: Optional[np.ndarray],
    title: str,
    *,
    fidelity: Optional[float] = None,
) -> None:
    """Plot one qubit's 2x2 marginal confusion matrix stored as ``conf[measured_bit, prepared_bit]``."""
    if conf is None:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    matrix = np.asarray(conf)
    ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", origin="upper")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("prepared")
    ax.set_ylabel("measured")

    for meas in range(2):
        for prep in range(2):
            probability = float(matrix[meas, prep])
            ax.text(
                prep,
                meas,
                f"{100 * probability:.2f}%",
                ha="center",
                va="center",
                color="white" if probability > 0.6 else "black",
            )

    ax.set_title(title if fidelity is None else f"{title}\nF = {100 * fidelity:.2f}%")


def plot_marginal_confusion_matrices_grid(
    names: Sequence[str],
    marginal_by_name: Dict[str, np.ndarray],
    fidelity_by_name: Optional[Dict[str, float]] = None,
    *,
    num_cols: int = 4,
    panel_size: float = 3.0,
) -> Figure:
    """Plot single-qubit marginal confusion matrices on a regular subplot grid."""
    num_qubits = len(names)
    num_cols = max(1, min(num_cols, num_qubits))
    num_rows = int(np.ceil(num_qubits / num_cols))
    fig = plt.figure(figsize=(panel_size * num_cols, panel_size * num_rows))

    for idx, name in enumerate(names):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
        fidelity = fidelity_by_name.get(name) if fidelity_by_name else None
        plot_marginal_confusion_matrix_on_axes(ax, marginal_by_name.get(name), name, fidelity=fidelity)

    fig.suptitle("Single-qubit marginals of the joint readout")
    fig.tight_layout()
    return fig


def plot_confusion_matrices_grid(
    target_names: Sequence[str],
    state_labels: Sequence[str],
    matrix_by_name: Dict[str, np.ndarray],
    title_fn: Callable[[str], str],
    *,
    num_cols: int = 3,
    panel_size: float = 5.0,
    annotate_cells: bool = True,
    text_fontsize: int = 10,
    is_difference: bool = False,
    cmap: Optional[str] = None,
    show_colorbar: bool = True,
) -> Figure:
    """Plot one confusion-matrix figure on a regular subplot grid."""
    num_targets = len(target_names)
    num_cols = max(1, min(num_cols, num_targets))
    num_rows = int(np.ceil(num_targets / num_cols))
    fig = plt.figure(figsize=(panel_size * num_cols, panel_size * num_rows))
    outer_grid = fig.add_gridspec(num_rows, num_cols)

    # Non-difference panels draw two colour bars (see `_plot_dual_colormap`); matplotlib's
    # automatic colorbar sizing doesn't guarantee they come out the same size when both are
    # attached to the same axes, so each panel's cell is pre-split here into
    # [heatmap | diagonal bar | off-diagonal bar], each colour bar getting an identical share of
    # the cell. The difference panel only ever needs one bar, which stays a simple, well-behaved
    # `ax=ax` colorbar (handled inside `plot_confusion_matrix_on_axes`).
    needs_dual_colorbars = show_colorbar and not is_difference
    panel_colorbars: List[Tuple[Axes, Tuple[Axes, Axes]]] = []

    for idx, name in enumerate(target_names):
        row, col = divmod(idx, num_cols)
        cell = outer_grid[row, col]
        colorbar_axes = None
        if needs_dual_colorbars:
            inner = cell.subgridspec(1, 3, width_ratios=[26, 1, 1], wspace=0.5)
            ax = fig.add_subplot(inner[0, 0])
            colorbar_axes = (fig.add_subplot(inner[0, 1]), fig.add_subplot(inner[0, 2]))
            panel_colorbars.append((ax, colorbar_axes))
        else:
            ax = fig.add_subplot(cell)

        plot_confusion_matrix_on_axes(
            ax,
            matrix_by_name.get(name),
            state_labels,
            title_fn(name),
            is_difference=is_difference,
            annotate_cells=annotate_cells,
            text_fontsize=text_fontsize,
            show_colorbar=show_colorbar,
            cmap=cmap,
            colorbar_axes=colorbar_axes,
        )

    fig.tight_layout()

    # tight_layout shrinks each heatmap axes to leave room for its own title/tick labels, but
    # (being a separate axes sharing only the gridspec row) leaves the colour bars at the full,
    # untrimmed row height -- taller than the heatmap they belong to. Snap them back to the
    # heatmap's final vertical extent.
    fig.canvas.draw()
    for ax, (cax_diagonal, cax_off) in panel_colorbars:
        ax_pos = ax.get_position()
        for cax in (cax_diagonal, cax_off):
            cax_pos = cax.get_position()
            cax.set_position([cax_pos.x0, ax_pos.y0, cax_pos.width, ax_pos.height])

    return fig
