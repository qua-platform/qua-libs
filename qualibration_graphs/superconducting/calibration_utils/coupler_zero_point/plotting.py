"""Plotting functions for coupler zero-point calibration.

This module provides visualization functions for displaying raw data and
fit results from coupler zero-point calibration experiments.
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: list,
    fits: Optional[Dict] = None,
) -> Dict[str, Figure]:
    """Plot raw 2D maps with fit overlays for all qubit pairs.

    Subplots are arranged using :class:`~calibration_utils.pair_grid.QubitPairGrid`,
    which positions each coupler between its two qubits on a doubled integer grid.
    Separate figures are returned for the target and control readouts.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset containing state or I quadrature data.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).
    fits : dict, optional
        Fit results keyed by qubit-pair name (``fit_results`` from analysis).

    Returns
    -------
    dict[str, Figure]
        Figures keyed by ``"target"`` and ``"control"``.
    """
    pair_by_name = {qp.name: qp for qp in qubit_pairs}
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, Figure] = {}

    for state_type in ("target", "control"):
        grid = QubitPairGrid(grid_names, pair_names)
        for ax, qubit in grid_iter(grid):
            qp_name = qubit["qubit"]
            fit_data = fits.get(qp_name) if fits else None
            plot_individual_data_with_fit(
                ax,
                ds,
                qubit,
                fit_data,
                data_var=state_type,
                qubit_pair_obj=pair_by_name.get(qp_name),
            )

        grid.fig.suptitle(f"Coupler zero point ({state_type})")
        grid.fig.tight_layout()
        figures[state_type] = grid.fig

    return figures


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: Optional[Dict] = None,
    data_var: str = "target",
    qubit_pair_obj=None,
):
    """Plot individual qubit-pair data on a given axis with optional fit overlays.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        Mapping from ``grid_iter`` containing the qubit-pair name under ``"qubit"``.
    fit : dict, optional
        Fit results for this qubit pair.
    data_var : str, optional
        Which qubit state to plot: ``"target"`` or ``"control"``.
    qubit_pair_obj : optional
        Qubit pair object used to annotate the current coupler idle offset.
    """
    qubit_pair = qubit["qubit"]

    if data_var == "control":
        if "state_control" in ds:
            data = ds.state_control.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_control.sel(qubit_pair=qubit_pair)
    else:
        if "state_target" in ds:
            data = ds.state_target.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_target.sel(qubit_pair=qubit_pair)

    data.assign_coords(
        {
            "qubit_flux_mV": 1e3 * data.qubit_flux_full,
            "coupler_flux_mV": 1e3 * data.coupler_flux_full,
        }
    ).plot(ax=ax, x="qubit_flux_mV", y="coupler_flux_mV", cmap="viridis")

    has_legend = False
    if fit and fit.get("success"):
        ax.axhline(
            1e3 * fit["optimal_coupler_flux"],
            color="red",
            lw=1.5,
            ls="--",
            label="Optimal coupler flux",
        )
        ax.axvline(
            1e3 * fit["optimal_qubit_flux"],
            color="red",
            lw=1.5,
            ls="--",
            label="Optimal qubit flux",
        )
        has_legend = True
        ax.set_title(f"{qubit_pair} ({data_var})")
    elif fit is not None:
        ax.set_title(f"{qubit_pair} ({data_var}) - Fit failed")
    else:
        ax.set_title(f"{qubit_pair} ({data_var})")

    if qubit_pair_obj is not None:
        idle_offset_mV = 1e3 * qubit_pair_obj.coupler.decouple_offset
        if np.isfinite(idle_offset_mV):
            ax.axhline(
                idle_offset_mV,
                color="blue",
                lw=0.5,
                ls="--",
                label="Current decouple offset",
            )
            has_legend = True

    if has_legend:
        ax.legend(fontsize=8)

    if "detuning" in ds:
        try:
            detuning_data = ds.sel(qubit_pair=qubit_pair).detuning.values * 1e-6
            flux_qubit_data = (1e3 * data.qubit_flux_full.values).ravel()
            detuning_data = detuning_data.ravel()

            order = np.argsort(flux_qubit_data)
            x_sorted = flux_qubit_data[order]
            y_sorted = detuning_data[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]

            if x_unique.size >= 2:

                def flux_to_detuning(x):
                    return np.interp(x, x_unique, y_unique)

                def detuning_to_flux(y):
                    return np.interp(y, y_unique, x_unique)

                sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
                sec_ax.set_xlabel("Detuning [MHz]")
        except (ValueError, TypeError, IndexError):
            pass

    ax.set_xlabel("Qubit flux [mV]")
    ax.set_ylabel("Coupler flux [mV]")
