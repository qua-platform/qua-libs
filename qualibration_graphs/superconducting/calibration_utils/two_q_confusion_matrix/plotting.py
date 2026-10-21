"""Plotting module for two-qubit readout confusion matrix calibration."""

from typing import Dict

from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.common_utils.confusion_matrix.plotting import (
    confusion_difference_panel_title,
    confusion_matrix_figure_suptitle,
    diff_confusion_matrices,
    get_state_labels,
    joint_measured_panel_title,
    plot_confusion_matrix_on_axes,
    tensor_product_panel_title,
)
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_STATE_LABELS = get_state_labels(2)
_FIGURE_SUPTITLE = confusion_matrix_figure_suptitle(2)


def _plot_pair_grid_figure(
    matrices: dict,
    qubit_pairs: list,
    node,
    panel_title_fn,
    *,
    is_difference: bool = False,
):
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    confusion_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(confusion_grid):
        qp_name = qubit["qubit"]
        plot_confusion_matrix_on_axes(
            ax,
            matrices.get(qp_name),
            _STATE_LABELS,
            panel_title_fn(qp_name),
            is_difference=is_difference,
            show_colorbar=is_difference,
        )
    confusion_grid.fig.suptitle(_FIGURE_SUPTITLE)
    confusion_grid.fig.tight_layout()
    return confusion_grid.fig


def plot_confusion_matrices(
    confusions: dict,
    qubit_pairs: list,
    node=None,
    kron_confs: dict | None = None,
) -> Dict[str, Figure]:
    """Plot 4x4 confusion matrices on a chip-topology grid."""
    joint_title = lambda qp_name: joint_measured_panel_title(qp_name, 2, node)
    figures = {
        "figure_confusion": _plot_pair_grid_figure(
            confusions,
            qubit_pairs,
            node,
            joint_title,
        )
    }
    if kron_confs is None:
        return figures

    target_names = [qp.name for qp in qubit_pairs]
    tensor_title = lambda qp_name: tensor_product_panel_title(qp_name, node)
    diff_title = lambda qp_name: confusion_difference_panel_title(qp_name, node)
    figures["figure_kron"] = _plot_pair_grid_figure(
        {name: kron_confs[name] for name in target_names if name in kron_confs},
        qubit_pairs,
        node,
        tensor_title,
    )
    figures["figure_diff"] = _plot_pair_grid_figure(
        diff_confusion_matrices(confusions, kron_confs, target_names),
        qubit_pairs,
        node,
        diff_title,
        is_difference=True,
    )
    return figures
