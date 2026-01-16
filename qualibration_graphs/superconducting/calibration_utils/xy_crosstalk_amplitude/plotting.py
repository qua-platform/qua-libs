from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
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
    figs = []
    for role in ds["role"]:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            plot_individual_data(ax, ds.sel(role=role, drop=True), qubit, fits.sel(qubit=qubit["qubit"], role=role))

        st_ = "Driven Qubit" if role == "d" else "Probed Qubit"
        grid.fig.suptitle(f"XY Crosstalk Amplitude - {st_}")
        # grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figs.append(grid.fig)
    return figs


def plot_individual_data(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    ds.sel(qubit).drop_vars("amp_ratio_p2d")[data].plot(
        ax=ax, add_colorbar=False, x="pulse_duration", y="amp_scaling", robust=True
    )
    ax.set_ylabel(f"Amplitude scaling")
    ax.set_xlabel("Pulse duration [ns]")

    # compute k for this qubit (constant scale factor)
    k = float(ds.loc[qubit].amp_ratio_p2d.values)

    # add right y-axis that converts scaling <-> ratio
    axr = ax.secondary_yaxis(
        "right",
        functions=(lambda s: s * k, lambda r: r / k)  # (left->right, right->left)
    )
    axr.set_ylabel("Amplitude ratio (probe/driven)")