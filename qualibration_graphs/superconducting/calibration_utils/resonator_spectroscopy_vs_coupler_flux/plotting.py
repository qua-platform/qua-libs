"""Plotting functions for resonator spectroscopy versus coupler flux calibration."""

from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import grid_iter
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    measured_qubits: List[AnyTransmon],
    fits: xr.Dataset = None,
) -> Figure:
    """Plot raw data with fitted curves for all qubit pairs.

    Subplots are arranged using ``QubitPairGrid``, which positions each
    coupler between its two qubits on a doubled integer grid.  This reflects
    the physical chip topology and leaves intentional gaps where the
    intermediate qubits would sit.

    The title of each subplot shows both the measured qubit and the coupler
    name so that duplicate measured-qubit situations are unambiguous.

    Parameters
    ----------
    ds:
        Raw dataset whose ``qubit`` coordinate contains coupler names.
    qubit_pairs:
        Ordered list of qubit-pair objects (same order as ``ds.qubit``).
    measured_qubits:
        Ordered list of the qubit being measured in each pair.
    fits:
        Fit dataset whose ``qubit`` coordinate also contains coupler names.
        If ``None``, only the raw data is plotted.
    """
    measured_by_coupler = {qp.name: q for qp, q in zip(qubit_pairs, measured_qubits)}

    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names, size=6)

    for ax, qubit in grid_iter(grid):
        coupler_name = qubit["qubit"]
        q = measured_by_coupler[coupler_name]
        fit_q = fits.sel(qubit=coupler_name) if fits is not None else None
        plot_individual_raw_data_with_fit(ax, ds.sel(qubit=coupler_name), fit_q)
        ax.set_title(f"Measured qubit: {q.name}\nCoupler: {coupler_name}")

    grid.fig.suptitle("Resonator spectroscopy vs coupler flux")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds_q: xr.Dataset, fit_q: xr.Dataset = None) -> None:
    """Plot data for a individual coupler qubit on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds_q : xr.Dataset
        Dataset already selected for one coupler.
    fit_q : xr.Dataset
        Fit dataset already selected for the same coupler.
    """
    ax2 = ax.twiny()

    ds_q.assign_coords(freq_GHz=ds_q.full_freq / 1e9).IQ_abs.plot(
        ax=ax2,
        add_colorbar=False,
        x="attenuated_current",
        y="freq_GHz",
        robust=True,
    )
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Freq (GHz)")
    ax2.set_title("")
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    ds_q.assign_coords(freq_GHz=ds_q.full_freq / 1e9).IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="flux_bias",
        y="freq_GHz",
        robust=True,
    )

    if fit_q is not None and fit_q.fit_results.success.values:
        ax.axvline(
            fit_q.fit_results.idle_offset,
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="idle offset",
        )
        ax.axvline(
            fit_q.fit_results.flux_min,
            linestyle="dashed",
            linewidth=2,
            color="orange",
            label="min offset",
        )
        ax.plot(
            fit_q.fit_results.idle_offset.values,
            fit_q.fit_results.sweet_spot_frequency.values * 1e-9,
            "r*",
            markersize=10,
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Coupler flux (V)")
    ax.set_ylabel("Freq (GHz)")
