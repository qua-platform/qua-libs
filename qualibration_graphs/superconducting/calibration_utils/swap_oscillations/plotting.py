import xarray as xr
from typing import List
import matplotlib.pyplot as plt
from qualibration_libs.plotting import QubitGrid
from quam_builder.architecture.superconducting.qubit import AnyTransmon
import numpy as np
from typing import List
import xarray as xr
from matplotlib.axes import Axes

from qualibration_libs.plotting import QubitGrid, grid_iter

def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset):
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
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))
    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs

        # Try to get fit data for this qubit pair, handle if missing
        try:
            fit_data = fits.sel(qubit_pair=qp.id) if fits is not None else None
        except (KeyError, ValueError):
            # If this qubit pair is not in the fit results, set fit_data to None
            fit_data = None

        plot_individual_data_with(ax, ds, qp.id, fit_data)

    fig.suptitle("SWAP oscillations")
    fig.set_size_inches(15, 9)
    fig.tight_layout()

    return fig

def plot_individual_data_with(ax: Axes, ds: xr.Dataset, qubit_pair: str, fit: xr.Dataset = None):
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

    if hasattr(ds, "state_target"):
        # If the dataset has 'state_target', use it for plotting
        data = ds.state_target
    else:
        data = ds.I_target

    qp_name = qp_info["qubit"]
    qp = next(qp for qp in qubit_pairs if qp.name == qp_name)
    fp = fit_results[qp_name]

    chevron = ds.sel(qubit=qp_name).state_control.assign_coords(
        detuning_MHz=1e-6 * ds.detuning.sel(qubit=qp_name)
    )
    plot = chevron.plot(ax=ax, x="idle_time", y="detuning_MHz", add_colorbar=False)
    plt.colorbar(plot, ax=ax, orientation="horizontal", pad=0.2)

    if fp.success:
        ax.axhline(fp.detuning * 1e-6, linestyle='--', color='k')
        ax.axvline(fp.optimal_length, linestyle='--', color='k')
        f_eff = np.sqrt(fp.J ** 2 + (ds.detuning.sel(qubit=qp_name) - fp.detuning) ** 2)
        for n in range(1, 10):
            ax.plot(n * 0.5 / f_eff * 1e9, ds.detuning.sel(qubit=qp_name) * 1e-6, 'r-', lw=0.3)

