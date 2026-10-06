"""Plotting module for N-qubit readout confusion matrix calibration."""

from typing import Dict, List

from matplotlib.figure import Figure

from calibration_utils.common_utils.confusion_matrix import (
    compute_marginal_confusion_matrices,
    marginal_assignment_fidelity,
)
from calibration_utils.common_utils.confusion_matrix.plotting import (
    annotation_style,
    confusion_difference_panel_title,
    confusion_matrix_figure_suptitle,
    diff_confusion_matrices,
    get_state_labels,
    joint_measured_panel_title,
    plot_confusion_matrices_grid,
    plot_marginal_confusion_matrices_grid,
    tensor_product_panel_title,
)

from .qubit_groups import QubitGroup


def _marginal_figure(confusions: dict, qubit_groups: List[QubitGroup]) -> Figure:
    """Build the single-qubit marginal confusion matrix grid across every qubit group.

    Qubit names are disambiguated with their group name whenever more than one group is
    configured, since the same qubit can appear in several groups.
    """
    disambiguate = len(qubit_groups) > 1
    names: List[str] = []
    marginal_by_name = {}
    fidelity_by_name = {}

    for qg in qubit_groups:
        qubit_names = [q.name for q in qg.qubits]
        group_marginals = compute_marginal_confusion_matrices(confusions[qg.name], qubit_names)
        for qubit_name in qubit_names:
            label = f"{qubit_name} ({qg.name})" if disambiguate else qubit_name
            names.append(label)
            marginal_by_name[label] = group_marginals[qubit_name]
            fidelity_by_name[label] = marginal_assignment_fidelity(group_marginals[qubit_name])

    return plot_marginal_confusion_matrices_grid(names, marginal_by_name, fidelity_by_name)


def plot_confusion_matrices(
    confusions: dict,
    kron_confs: dict,
    qubit_groups: List[QubitGroup],
    node=None,
) -> Dict[str, Figure]:
    """Plot direct, Kronecker, difference, and single-qubit marginal confusion matrices."""
    num_qubits = qubit_groups[0].num_qubits
    state_labels = get_state_labels(num_qubits)
    target_names = [qg.name for qg in qubit_groups]
    annotate_cells, text_fontsize = annotation_style(num_qubits)
    figure_suptitle = confusion_matrix_figure_suptitle(num_qubits)
    grid_kwargs = {
        "annotate_cells": annotate_cells,
        "text_fontsize": text_fontsize,
        "figure_suptitle": figure_suptitle,
    }

    figures = {
        "figure_confusion": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            confusions,
            lambda group_name: joint_measured_panel_title(group_name, num_qubits, node),
            **grid_kwargs,
        ),
        "figure_kron": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            kron_confs,
            lambda group_name: tensor_product_panel_title(group_name, node),
            **grid_kwargs,
        ),
        "figure_diff": plot_confusion_matrices_grid(
            target_names,
            state_labels,
            diff_confusion_matrices(confusions, kron_confs, target_names),
            lambda group_name: confusion_difference_panel_title(group_name, node),
            is_difference=True,
            cmap="RdBu",
            **grid_kwargs,
        ),
    }

    # A single qubit's marginal is just its own 2x2 confusion matrix, already shown above.
    if num_qubits >= 2:
        figures["figure_marginals"] = _marginal_figure(confusions, qubit_groups)

    return figures
