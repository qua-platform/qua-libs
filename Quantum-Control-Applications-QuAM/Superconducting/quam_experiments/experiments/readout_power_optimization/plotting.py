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
    # for ax, qubit in grid_iter(grid):
    #     fit_res.loc[qubit].plot(ax=ax, x="readout_amp", hue="result", add_legend=False)
    #     ax.axvline(best_amp[qubit["qubit"]], color="k", linestyle="dashed")
    #     ax.set_xlabel("Relative power")
    #     ax.set_ylabel("Fidelity / outliers")
    #     ax.set_title(qubit["qubit"])
    # grid.fig.suptitle("Assignment fidelity and non-outlier probability")
    #
    # plt.tight_layout()
    # plt.show()
    # node.results["figure_assignment_fid"] = grid.fig
    #
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    # for ax, qubit in grid_iter(grid):
    #     ds_q = best_data[qubit["qubit"]]
    #     qn = qubit["qubit"]
    #     ax.plot(
    #         1e3
    #         * (
    #                 ds_q.I.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
    #                 - ds_q.Q.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
    #         ),
    #         1e3
    #         * (
    #                 ds_q.I.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
    #                 + ds_q.Q.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
    #         ),
    #         ".",
    #         alpha=0.1,
    #         label="Ground",
    #         markersize=1,
    #     )
    #     ax.plot(
    #         1e3
    #         * (
    #                 ds_q.I.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
    #                 - ds_q.Q.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
    #         ),
    #         1e3
    #         * (
    #                 ds_q.I.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
    #                 + ds_q.Q.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
    #         ),
    #         ".",
    #         alpha=0.1,
    #         label="Excited",
    #         markersize=1,
    #     )
    #     ax.axvline(
    #         1e3 * node.results["results"][qn]["rus_threshold"],
    #         color="k",
    #         linestyle="--",
    #         lw=0.5,
    #         label="RUS Threshold",
    #     )
    #     ax.axvline(
    #         1e3 * node.results["results"][qn]["threshold"],
    #         color="r",
    #         linestyle="--",
    #         lw=0.5,
    #         label="Threshold",
    #     )
    #     ax.axis("equal")
    #     ax.set_xlabel("I [mV]")
    #     ax.set_ylabel("Q [mV]")
    #     ax.set_title(qubit["qubit"])
    #
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    # plt.tight_layout()
    # node.results["figure_IQ_blobs"] = grid.fig
    #
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    # for ax, qubit in grid_iter(grid):
    #     confusion = node.results["results"][qubit["qubit"]]["confusion_matrix"]
    #     ax.imshow(confusion)
    #     ax.set_xticks([0, 1])
    #     ax.set_yticks([0, 1])
    #     ax.set_xticklabels(labels=["|g>", "|e>"])
    #     ax.set_yticklabels(labels=["|g>", "|e>"])
    #     ax.set_ylabel("Prepared")
    #     ax.set_xlabel("Measured")
    #     ax.text(
    #         0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k"
    #     )
    #     ax.text(
    #         1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w"
    #     )
    #     ax.text(
    #         0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w"
    #     )
    #     ax.text(
    #         1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k"
    #     )
    #     ax.set_title(qubit["qubit"])
    #
    # grid.fig.suptitle("g.s. and e.s. fidelity")
    # plt.tight_layout()
    # plt.show()
    # node.results["figure_fidelities"] = grid.fig
