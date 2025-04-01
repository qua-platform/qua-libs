from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qualibration_libs.plot_utils import QubitGrid, grid_iter
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


def plot_raw_data():
    pass
    # if node.parameters.plot_raw:
    #     fig, axes = plt.subplots(
    #         ncols=num_qubits,
    #         nrows=len(ds.amplitude),
    #         sharex=False,
    #         sharey=False,
    #         squeeze=False,
    #         figsize=(5 * num_qubits, 5 * len(ds.amplitude)),
    #     )
    #     for amplitude, ax1 in zip(ds.amplitude, axes):
    #         for q, ax2 in zip(list(qubits), ax1):
    #             ds_q = ds.sel(qubit=q.name, amplitude=amplitude)
    #             ax2.plot(
    #                 ds_q.I.sel(state=0),
    #                 ds_q.Q.sel(state=0),
    #                 ".",
    #                 alpha=0.2,
    #                 label="Ground",
    #                 markersize=2,
    #             )
    #             ax2.plot(
    #                 ds_q.I.sel(state=1),
    #                 ds_q.Q.sel(state=1),
    #                 ".",
    #                 alpha=0.2,
    #                 label="Excited",
    #                 markersize=2,
    #             )
    #             ax2.set_xlabel("I")
    #             ax2.set_ylabel("Q")
    #             ax2.set_title(f"{q.name}, {float(amplitude)}")
    #             ax2.axis("equal")
    #     plt.show()
    #     node.results["figure_raw_data"] = fig


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
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    #     for ax, qubit in grid_iter(grid):
    #         ds.loc[qubit].adc_single_runI.plot(ax=ax, x="time", label="I", color="b")
    #         ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="time", label="Q", color="r")
    #         ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
    #         ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    #         ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    #         ax.fill_between(
    #             range(ds.sizes["time"]),
    #             -0.5,
    #             0.5,
    #             color="grey",
    #             alpha=0.2,
    #             label="ADC Range",
    #         )
    #         ax.set_xlabel("Time [ns]")
    #         ax.set_ylabel("Readout amplitude [mV]")
    #         ax.set_title(qubit["qubit"])
    #     grid.fig.suptitle("Single run")
    #     plt.tight_layout()
    #     plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    #     node.results["adc_single_run"] = grid.fig
    #
    #     # Averaged run
    #     grid = QubitGrid(ds, [q.grid_location for q in qubits])
    #     for ax, qubit in grid_iter(grid):
    #         ds.loc[qubit].adcI.plot(ax=ax, x="time", label="I", color="b")
    #         ds.loc[qubit].adcQ.plot(ax=ax, x="time", label="Q", color="r")
    #         ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
    #         ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    #         ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    #         ax.set_xlabel("Time [ns]")
    #         ax.set_ylabel("Readout amplitude [mV]")
    #         ax.set_title(qubit["qubit"])
    #     grid.fig.suptitle("Averaged run")
    #     plt.tight_layout()
    #     plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    #     node.results["adc_averaged"] = grid.fig


def plot_adc_single_runs(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].adc_single_runI.plot(ax=ax, x="time", label="I", color="b")
        ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="time", label="Q", color="r")
        ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
        ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
        ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
        ax.fill_between(
            range(ds.sizes["time"]),
            -0.5,
            0.5,
            color="grey",
            alpha=0.2,
            label="ADC Range",
        )
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Readout amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Single run")
    plt.tight_layout()
    plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))

    return grid.fig


def plot_adc_averaged_runs(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].adcI.plot(ax=ax, x="time", label="I", color="b")
        ds.loc[qubit].adcQ.plot(ax=ax, x="time", label="Q", color="r")
        ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
        ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
        ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Readout amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Averaged run")
    plt.tight_layout()
    plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))

    return grid.fig
