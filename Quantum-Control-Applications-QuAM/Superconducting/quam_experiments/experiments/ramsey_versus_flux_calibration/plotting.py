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
    # grid = QubitGrid(ds, grid_names)
    # for ax, qubit in grid_iter(grid):
    #     ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    #     ax.set_title(qubit["qubit"])
    #     ax.set_xlabel("Idle_time (uS)")
    #     ax.set_ylabel(" Flux (V)")
    # grid.fig.suptitle("Ramsey freq. Vs. flux")
    # plt.tight_layout()
    # plt.show()
    # node.results["figure_raw"] = grid.fig
    #
    # grid = QubitGrid(ds, grid_names)
    # for ax, qubit in grid_iter(grid):
    #     fitted_freq = (
    #             fitvals.sel(qubit=qubit["qubit"], degree=2).polyfit_coefficients * flux ** 2
    #             + fitvals.sel(qubit=qubit["qubit"], degree=1).polyfit_coefficients * flux
    #             + fitvals.sel(qubit=qubit["qubit"], degree=0).polyfit_coefficients
    #     )
    #     frequency.sel(qubit=qubit["qubit"]).plot(marker=".", linewidth=0, ax=ax)
    #     ax.plot(flux, fitted_freq)
    #     ax.set_title(qubit["qubit"])
    #     ax.set_xlabel(" Flux (V)")
    #     print(
    #         f"The quad term for {qubit['qubit']} is {a[qubit['qubit']] / 1e9:.3f} GHz/V^2"
    #     )
    #     print(
    #         f"Flux offset for {qubit['qubit']} is {flux_offset[qubit['qubit']] * 1e3:.1f} mV"
    #     )
    #     print(
    #         f"Freq offset for {qubit['qubit']} is {freq_offset[qubit['qubit']] / 1e6:.3f} MHz"
    #     )
    #     print()
    # grid.fig.suptitle("Ramsey freq. Vs. flux")
    # plt.tight_layout()
    # plt.show()
    # node.results["figure"] = grid.fig
