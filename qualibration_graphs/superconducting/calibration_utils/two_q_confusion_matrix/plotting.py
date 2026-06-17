"""Plotting module for two-qubit readout confusion matrix calibration."""

from typing import Dict, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_STATE_LABELS = ["00", "01", "10", "11"]


def plot_confusion_matrices(
    confusions: Dict[str, np.ndarray],
    qubit_pairs: list,
    node=None,
) -> Dict[str, Figure]:
    """Plot 4x4 confusion matrices on a chip-topology grid.

    Parameters
    ----------
    confusions : dict[str, np.ndarray]
        Mapping from qubit pair name to 4x4 confusion matrix stored as
        ``conf[measured, prepared]`` (display uses the transpose).
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    node : optional
        Node for metadata (reset_type) in subplot titles.

    Returns
    -------
    dict[str, Figure]
        ``"figure_confusion"`` contains all pair confusion matrices.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    confusion_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(confusion_grid):
        qp_name = qubit["qubit"]
        plot_individual_confusion_matrix(ax, confusions.get(qp_name), qp_name, node=node)
    confusion_grid.fig.suptitle("Two-qubit readout confusion matrix")
    confusion_grid.fig.tight_layout()
    return {"figure_confusion": confusion_grid.fig}


def plot_individual_confusion_matrix(
    ax: Axes,
    conf: Optional[np.ndarray],
    qp_name: str,
    node=None,
) -> None:
    """Plot one qubit-pair 4x4 confusion matrix.

    Stored matrices are ``conf[measured, prepared]``; the heatmap shows
    ``conf.T[prepared, measured]`` with prepared on the y-axis and measured on the x-axis.
    """
    if conf is None:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    display = np.asarray(conf).T  # [prepared, measured] for plotting
    ax.pcolormesh(_STATE_LABELS, _STATE_LABELS, display)
    for prep in range(4):
        for meas in range(4):
            color = "k" if prep == meas else "w"
            ax.text(
                meas,
                prep,
                f"{100 * display[prep, meas]:.1f}%",
                ha="center",
                va="center",
                color=color,
            )
    ax.set_ylabel("prepared")
    ax.set_xlabel("measured")

    title = qp_name
    if node is not None:
        reset_type = getattr(node.parameters, "reset_type", None)
        if reset_type is not None:
            title = f"{qp_name}\nreset type = {reset_type}"
    ax.set_title(title)
