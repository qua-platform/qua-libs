"""Plotting module for N-qubit readout confusion matrix calibration."""

from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import QubitGroup


def _plot_matrix_figure(
    qubit_groups: List[QubitGroup],
    state_labels: List[str],
    matrix_by_group: Dict[str, np.ndarray],
    title_fn: Callable[[str, int], str],
    num_qubits: int,
    annotate_cells: bool,
    text_fontsize: int,
    cmap: Optional[str] = None,
    is_difference: bool = False,
) -> Figure:
    """Plot one matrix figure for all qubit groups using shared styling."""
    num_groups = len(qubit_groups)
    num_cols = 3
    num_rows = int(np.ceil(num_groups / num_cols))
    num_states = len(state_labels)
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for idx, qg in enumerate(qubit_groups):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
        matrix = matrix_by_group[qg.name]

        if is_difference:
            max_abs = np.max(np.abs(matrix))
            mesh = ax.pcolormesh(
                state_labels,
                state_labels,
                matrix,
                cmap=cmap or "RdBu",
                vmin=-max_abs,
                vmax=max_abs,
            )
        else:
            mesh = ax.pcolormesh(state_labels, state_labels, matrix, cmap=cmap)

        if annotate_cells:
            for prep in range(num_states):
                for meas in range(num_states):
                    val = matrix[prep, meas]
                    if is_difference and abs(val) <= 0.01:
                        continue
                    if is_difference:
                        color = "k" if abs(val) < 0.5 * max_abs else "w"
                    else:
                        color = "k" if prep == meas else "w"
                    ax.text(
                        meas,
                        prep,
                        f"{100 * val:.1f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=text_fontsize,
                    )

        ax.set_ylabel("prepared")
        ax.set_xlabel("measured")
        ax.set_title(title_fn(qg.name, num_qubits))
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    confusions: Dict[str, np.ndarray],
    kron_confs: Dict[str, np.ndarray],
    qubit_groups: List[QubitGroup],
    state_labels: List[str],
    node=None,
) -> Dict[str, Figure]:
    """Plot direct, Kronecker, and difference confusion matrices."""
    num_qubits = qubit_groups[0].num_qubits
    annotate_cells = num_qubits <= 3
    if num_qubits <= 2:
        text_fontsize = 10
    elif num_qubits == 3:
        text_fontsize = 8
    elif num_qubits == 4:
        text_fontsize = 6
    else:
        text_fontsize = 4

    reset_type = getattr(getattr(node, "parameters", None), "reset_type", None)

    def _title(prefix: str, group_name: str, nq: int) -> str:
        title = f"{prefix} {group_name} ({nq}Q)"
        if reset_type is not None:
            title = f"{title}\nreset type = {reset_type}"
        return title

    diff_confs = {qg.name: confusions[qg.name] - kron_confs[qg.name] for qg in qubit_groups}

    return {
        "figure_confusion": _plot_matrix_figure(
            qubit_groups,
            state_labels,
            confusions,
            lambda group_name, nq: _title("Confusion matrix", group_name, nq),
            num_qubits,
            annotate_cells,
            text_fontsize,
        ),
        "figure_kron": _plot_matrix_figure(
            qubit_groups,
            state_labels,
            kron_confs,
            lambda group_name, nq: _title("Kronecker confusion matrix", group_name, nq),
            num_qubits,
            annotate_cells,
            text_fontsize,
        ),
        "figure_diff": _plot_matrix_figure(
            qubit_groups,
            state_labels,
            diff_confs,
            lambda group_name, nq: _title("Difference (Direct - Kron)", group_name, nq),
            num_qubits,
            annotate_cells,
            text_fontsize,
            cmap="RdBu",
            is_difference=True,
        ),
    }
