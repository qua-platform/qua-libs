"""Plotting for three-tone coupler spectroscopy with flux pulse (22a)."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import xarray as xr
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from qualibration_libs.plotting import grid_iter

from .analysis import FitResults


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs,
    fit_results: Dict[str, Any],
) -> plt.Figure:
    """Plot target response vs coupler drive frequency with the extracted resonance marked."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    use_state = "state" in ds
    for ax, qp_info in grid_iter(grid):
        pair_name = qp_info["qubit"]
        ds_g = ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9)
        if use_state:
            ds_g.sel(qubit=pair_name).state.plot(ax=ax, x="freq_GHz")
            ax.set_ylabel("Target qubit state")
        else:
            ds_g.sel(qubit=pair_name).I.plot(ax=ax, x="freq_GHz")
            ax.set_ylabel("I")

        fit = fit_results.get(pair_name)
        if fit is not None and fit.get("success", getattr(fit, "success", False)):
            freq_ghz = (
                fit["coupler_frequency_hz"] * 1e-9
                if isinstance(fit, dict)
                else fit.coupler_frequency_hz * 1e-9
            )
            ax.axvline(freq_ghz, color="red", linestyle="--", alpha=0.5)
            flux_mv = (
                fit["coupler_flux_v"] * 1e3 if isinstance(fit, dict) else fit.coupler_flux_v * 1e3
            )
            ax.set_title(f"{pair_name}\nCoupler flux = {flux_mv:.2f} mV")
        else:
            ax.set_title(pair_name)
        ax.set_xlabel("Frequency (GHz)")

    grid.fig.suptitle("Three-tone coupler spectroscopy (flux pulse)")
    grid.fig.tight_layout()
    return grid.fig
