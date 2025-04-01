from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plot_utils import QubitGrid, grid_iter
from quam_experiments.analysis.fit import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Qubit spectroscopy (rotated 'I' quadrature + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    pass
    # grid_names = [q.grid_location for q in qubits]
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    # for ax, qubit in grid_iter(grid):
    #     da_state_qubit = da_state.sel(qubit=qubit["qubit"])
    #     da_state_std = ds["state"].std(dim="sequence").sel(
    #         qubit=qubit["qubit"]
    #     ) / np.sqrt(ds.sequence.size)
    #     ax.errorbar(
    #         da_state_qubit.m,
    #         da_state_qubit,
    #         yerr=da_state_std,
    #         fmt=".",
    #         capsize=2,
    #         elinewidth=0.5,
    #     )
    #     ax.grid("all")
    #     m = da_state.m.values
    #     ax.set_title(qubit["qubit"], pad=22)
    #     ax.set_xlabel("Circuit depth")
    #     fit_dict = {
    #         k: da_fit.sel(**qubit).sel(fit_vals=k).values
    #         for k in da_fit.fit_vals.values
    #     }
    #     ax.plot(m, decay_exp(m, **fit_dict), "r--", label="fit")
    #     ax.text(
    #         0.0,
    #         1.07,
    #         f"RB fidelity = {1 - EPG.sel(**qubit).values:.5f}",
    #         transform=ax.transAxes,
    #     )
    # plt.tight_layout()
    # plt.show()
    # node.results["figure"] = grid.fig
