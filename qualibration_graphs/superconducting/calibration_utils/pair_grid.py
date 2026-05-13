"""Shared grid utilities for qubit-pair (coupler) calibration plots.

Ported from flagship_qua/quam_libs/lib/plot_utils.py.  Provides
``QubitPairGrid`` which auto-computes a chip-topology subplot layout
from qubit grid locations, and ``grid_pair_names`` which extracts the
required metadata from qubit-pair objects.

Iteration over the resulting grid is done with the existing
``grid_iter`` from ``qualibration_libs.plotting``.
"""

import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def grid_pair_names(qubit_pairs) -> Tuple[List[str], List[str]]:
    """Return grid-location strings and pair names for a list of qubit pairs.

    Each grid-location string has the form
    ``"<control_grid_location>-<target_grid_location>"``
    (e.g. ``"0,0-1,0"``).

    Returns
    -------
    grid_names : list[str]
        One entry per pair, encoding both qubits' grid locations.
    qubit_pair_names : list[str]
        One entry per pair, the pair's ``.name`` attribute.
    """
    return (
        [f"{qp.qubit_control.grid_location}-{qp.qubit_target.grid_location}" for qp in qubit_pairs],
        [qp.name for qp in qubit_pairs],
    )


class QubitPairGrid:
    """Subplot grid that places coupler pairs at their physical chip positions.

    The layout is computed automatically: qubits sit on a doubled integer
    grid (even indices) and couplers sit between them (odd indices for
    nearest-neighbour pairs).  This produces the familiar staggered layout
    without any hardcoded position mapping.

    The resulting object exposes ``.fig``, ``.axes``, and ``.name_dicts``
    with the same structure as ``qualibration_libs.plotting.QubitGrid``,
    so it works with the existing ``grid_iter`` helper.

    Parameters
    ----------
    grid_names : list[str]
        Grid-location strings as returned by ``grid_pair_names``.
    qubit_pair_names : list[str]
        Pair name strings (used as ``qubit`` coordinate labels).
    size : int
        Size of each subplot in inches.
    """

    @staticmethod
    def _clean_up(input_string: str) -> str:
        return re.sub("[^0-9]", "", input_string)

    def _list_clean(self, list_input_string):
        return [self._clean_up(s) for s in list_input_string]

    def __init__(self, grid_names: list[str], qubit_pair_names: list[str], size: int = 4):
        qubit_indices = [
            (
                tuple(map(int, self._list_clean(gp.split("-")[0].split(",")))),
                tuple(map(int, self._list_clean(gp.split("-")[1].split(",")))),
            )
            for gp in grid_names
        ]

        row_diffs = [pair[1][0] - pair[0][0] for pair in qubit_indices]
        col_diffs = [pair[1][1] - pair[0][1] for pair in qubit_indices]

        # Place each coupler on a doubled grid so that qubits occupy even
        # positions and couplers fall in between.
        coupler_indices = [[2 * pair[0][1], 2 * pair[0][0]] for pair in qubit_indices]
        for k, (col_diff, row_diff) in enumerate(zip(col_diffs, row_diffs)):
            coupler_indices[k][0] += col_diff
            coupler_indices[k][1] += row_diff
        coupler_indices = [tuple(c) for c in coupler_indices]

        grid_row_idxs = [idx[0] for idx in coupler_indices]
        grid_col_idxs = [idx[1] for idx in coupler_indices]
        min_grid_row = min(grid_row_idxs)
        min_grid_col = min(grid_col_idxs)
        shape = (
            max(grid_row_idxs) - min_grid_row + 1,
            max(grid_col_idxs) - min_grid_col + 1,
        )

        figure, all_axes = plt.subplots(*shape, figsize=(shape[1] * size, shape[0] * size), squeeze=False)

        axes_grid = all_axes.reshape(shape)

        axes = []
        qubit_names = []

        for row, axis_row in enumerate(axes_grid):
            for col, ax in enumerate(axis_row):
                grid_row = row + min_grid_row
                grid_col = col + min_grid_col
                if (grid_row, grid_col) in coupler_indices:
                    axes.append(ax)
                    index = coupler_indices.index((grid_row, grid_col))
                    qubit_names.append(qubit_pair_names[index])
                else:
                    ax.axis("off")

        self.fig = figure
        self.all_axes = all_axes
        self.axes = [axes]
        self.name_dicts = [[{"qubit": name} for name in qubit_names]]
