"""Plotting module for N-qubit readout confusion matrix calibration."""

from typing import Dict, List

from matplotlib.figure import Figure

from calibration_utils.common_utils.confusion_matrix.plotting import (
    annotation_style,
    diff_confusion_matrices,
    get_state_labels,
    plot_confusion_matrices_grid,
    reset_type_subtitle,
)

from .qubit_groups import QubitGroup


def plot_confusion_matrices(
    confusions: dict,
    kron_confs: dict,
    qubit_groups: List[QubitGroup],
    node=None,
) -> Dict[str, Figure]:
    """Plot direct, Kronecker, and difference confusion matrices."""
    num_qubits = qubit_groups[0].num_qubits
    state_labels = get_state_labels(num_qubits)
    target_names = [qg.name for qg in qubit_groups]
    annotate_cells, text_fontsize = annotation_style(num_qubits)
    subtitle = reset_type_subtitle(node)
    grid_kwargs = {
        "annotate_cells": annotate_cells,
        "text_fontsize": text_fontsize,
    }

    def panel_title(prefix: str, group_name: str) -> str:
        return f"{prefix} {group_name} ({num_qubits}Q){subtitle}"

    return {
        "figure_confusion": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            confusions,
            lambda group_name: panel_title("Confusion matrix", group_name),
            **grid_kwargs,
        ),
        "figure_kron": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            kron_confs,
            lambda group_name: panel_title("Kronecker confusion matrix", group_name),
            **grid_kwargs,
        ),
        "figure_diff": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            diff_confusion_matrices(confusions, kron_confs, target_names),
            lambda group_name: panel_title("Difference (Direct - Kron)", group_name),
            is_difference=True,
            cmap="RdBu",
            **grid_kwargs,
        ),
    }
