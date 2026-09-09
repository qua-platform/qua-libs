"""Plotting helpers for readout confusion matrix calibrations."""

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
) -> None:
    """Plot one confusion matrix stored as ``conf[measured, prepared]``."""
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
        max_abs = np.max(np.abs(matrix))
        mesh = ax.pcolormesh(
            ticks,
            ticks,
            matrix,
            cmap=cmap or "RdBu",
            vmin=-max_abs,
            vmax=max_abs,
        )
    else:
        mesh = ax.pcolormesh(ticks, ticks, matrix, cmap=cmap)

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
                if is_difference:
                    color = "k" if abs(val) < 0.5 * max_abs else "w"
                else:
                    color = "k" if meas == prep else "w"
                ax.text(
                    prep,
                    meas,
                    f"{100 * val:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=text_fontsize,
                )

    ax.set_ylabel("measured" + (" state index" if use_integer_ticks else ""))
    ax.set_xlabel("prepared" + (" state index" if use_integer_ticks else ""))
    ax.set_title(title)
    if show_colorbar:
        ax.figure.colorbar(mesh, ax=ax)


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
    num_rows = int(np.ceil(num_targets / num_cols))
    fig = plt.figure(figsize=(panel_size * num_cols, panel_size * num_rows))

    for idx, name in enumerate(target_names):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
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
        )

    fig.tight_layout()
    return fig
