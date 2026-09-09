"""Plotting module for N-qubit readout confusion matrix calibration."""

from typing import Dict, List

from matplotlib.figure import Figure

from calibration_utils.common_utils.confusion_matrix import (
    compute_marginal_confusion_matrices,
    marginal_assignment_fidelity,
)
from calibration_utils.common_utils.confusion_matrix.plotting import (
    annotation_style,
    diff_confusion_matrices,
    get_state_labels,
    plot_confusion_matrices_grid,
    plot_marginal_confusion_matrices_grid,
    reset_type_subtitle,
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
    subtitle = reset_type_subtitle(node)
    grid_kwargs = {
        "annotate_cells": annotate_cells,
        "text_fontsize": text_fontsize,
    }

    def panel_title(prefix: str, group_name: str) -> str:
        return f"{prefix} {group_name} ({num_qubits}Q){subtitle}"

    figures = {
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

    # A single qubit's marginal is just its own 2x2 confusion matrix, already shown above.
    if num_qubits >= 2:
        figures["figure_marginals"] = _marginal_figure(confusions, qubit_groups)

    return figures
