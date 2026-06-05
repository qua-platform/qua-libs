from typing import List, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: list,
    fits: xr.Dataset,
) -> Figure:
    """Plot the CZ chevron 2D map for every qubit pair on a chip-topology grid.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset (must contain ``amp_full``, ``detuning``, and either
        ``state_target`` or ``I_target``).
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    fits : xr.Dataset
        Dataset containing fit parameters (``J``, ``f0``, ``a``, ``offset``,
        ``cz_len``, ``cz_amp``).

    Returns
    -------
    Figure
        Matplotlib figure with one chevron panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        try:
            fit_data = fits.sel(qubit_pair=qp_name) if fits is not None else None
        except (KeyError, ValueError):
            fit_data = None
        plot_individual_data_with_fit(ax, ds, qp_name, fit_data)

    grid.fig.suptitle("CZ Chevron")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qp_name: str,
    fit: Optional[xr.Dataset] = None,
):
    """Plot the chevron 2D map for one qubit pair with model contours and fit marker.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds : xr.Dataset
        Processed dataset with ``amp_full`` and ``detuning`` coordinates.
    qp_name : str
        Qubit pair name used to select data from ``ds``.
    fit : xr.Dataset, optional
        Fit parameters for this pair. When valid, the Rabi chevron model is
        overlaid as white contour lines and the optimal point is marked.
    """
    data = ds.state_target if hasattr(ds, "state_target") else ds.I_target
    data.sel(qubit_pair=qp_name).plot(y="amp_full", ax=ax)

    if fit is not None:
        try:
            cz_len = float(fit.cz_len)
            cz_amp = float(fit.cz_amp)

            if not (np.isnan(cz_len) or np.isnan(cz_amp)):
                ax.axvline(cz_len, color="red", ls="--", lw=1.2)
                ax.axhline(cz_amp, color="orange", ls="--", lw=1.2)
                ax.scatter(cz_len, cz_amp, color="red", marker="*", s=200, zorder=5, label="CZ point")
                ax.legend(fontsize=8)
                ax.set_title(f"{qp_name} — fit OK")
            else:
                ax.set_title(f"{qp_name} — fit failed")

        except (ValueError, TypeError, AttributeError, KeyError):
            ax.set_title(f"{qp_name} — fit failed")
    else:
        ax.set_title(qp_name)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Amplitude [V]")
