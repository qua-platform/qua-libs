"""Plotting for three-tone coupler spectroscopy vs coupler flux (22b)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from qualibration_libs.plotting import grid_iter


def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs) -> plt.Figure:
    """Plot 2D target response vs coupler flux and drive frequency."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)
    flux_coord = "flux" if "flux" in ds.dims else "coupler_flux"
    use_state = "state" in ds

    for ax, qp_info in grid_iter(grid):
        pair_name = qp_info["qubit"]
        ds_g = ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9)
        if use_state:
            ds_g.sel(qubit=pair_name).state.plot(ax=ax, y="freq_GHz", x=flux_coord)
        else:
            ds_g.sel(qubit=pair_name).IQ_abs.plot(ax=ax, y="freq_GHz", x=flux_coord)
        ax.set_title(pair_name)
        ax.set_ylabel("Frequency (GHz)")
        ax.set_xlabel("Coupler flux pulse (V)")

    grid.fig.suptitle("Three-tone coupler spectroscopy vs flux")
    grid.fig.tight_layout()
    return grid.fig
