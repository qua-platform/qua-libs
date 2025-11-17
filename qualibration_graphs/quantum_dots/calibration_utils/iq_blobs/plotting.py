from typing import List, Any
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_iq_blobs(ds: xr.Dataset, qubit_pairs: List[Any], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given quantum dot pairs.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pairs : list
        A list of quantum dot pairs to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each quantum dot pair.
    - Each subplot contains the raw data and the fitted curve.
    """
    # Calculate grid dimensions based on number of qubit pairs
    n_pairs = len(qubit_pairs)
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))

    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols)

    # Flatten axes array for easy iteration
    if n_pairs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each qubit pair
    for idx, qubit_pair in enumerate(qubit_pairs):
        ax = axes[idx]
        qubit_pair_dict = {"qubit_pair": qubit_pair.id}
        plot_individual_iq_blobs(ax, ds, qubit_pair_dict, fits.sel(qubit_pair=qubit_pair.id))

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    if n_pairs > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="lower center", ncol=2)
        leg.legend_handles[0].set_markersize(6)
        leg.legend_handles[1].set_markersize(6)

    fig.suptitle("Singlet and Triplet discriminators (rotated)")
    fig.tight_layout()
    return fig


def plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit_pair: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual quantum dot pair data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pair : dict[str, str]
        mapping to the quantum dot pair to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ax.plot(1e3 * fit.Ig_rot, 1e3 * fit.Qg_rot, ".", alpha=0.5, label="Singlet", markersize=1)
    ax.plot(
        1e3 * fit.Ie_rot,
        1e3 * fit.Qe_rot,
        ".",
        alpha=0.5,
        label="Triplet",
        markersize=1,
    )
    ax.axvline(
        1e3 * fit.I_threshold,
        color="k",
        linestyle="--",
        lw=0.5,
        label="I Threshold",
    )
    # ax.axvline(1e3 * fit.ge_threshold, color="r", linestyle="--", lw=0.5, label="Threshold")
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(f'{qubit_pair["qubit_pair"]}')


def plot_historams(ds: xr.Dataset, qubit_pairs: List[Any], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given quantum dot pairs.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pairs : list
        A list of quantum dot pairs to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each quantum dot pair.
    - Each subplot contains the raw data and the fitted curve.
    """
    # Calculate grid dimensions based on number of qubit pairs
    n_pairs = len(qubit_pairs)
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))

    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols)

    # Flatten axes array for easy iteration
    if n_pairs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each qubit pair
    for idx, qubit_pair in enumerate(qubit_pairs):
        ax = axes[idx]
        qubit_pair_dict = {"qubit_pair": qubit_pair.id}
        plot_individual_histograms(ax, ds, qubit_pair_dict, fits.sel(qubit_pair=qubit_pair.id))

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    if n_pairs > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)

    fig.suptitle("Singlet and Triplet histograms (rotated)")
    fig.tight_layout()
    return fig


def plot_individual_histograms(ax: Axes, ds: xr.Dataset, qubit_pair: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual quantum dot pair data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pair : dict[str, str]
        mapping to the quantum dot pair to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    - Plots histograms with fitted Barthel model curves using pre-computed densities.
    """
    # Combine rotated I data from both states
    y_combined = np.concatenate([fit.Ig_rot.values, fit.Ie_rot.values])

    # Plot histogram of combined data (in mV)
    ax.hist(1e3 * y_combined, bins=120, alpha=0.5, density=True, label="Data")

    # Plot pre-computed density curves if available
    # if 'density_grid' in ds and 'density_total' in ds:
    # Get pre-computed grid and densities (in mV)
    xs = fit.density_grid.values
    total = fit.density_total.values
    S_comp = fit.density_S.values
    T_no_comp = fit.density_T_no.values
    T_dec_comp = fit.density_T_dec.values

    # Plot fitted curves (convert to mV for plotting)
    ax.scatter(1e3 * xs, total, lw=2, label="Total (PPD mean)")
    ax.scatter(1e3 * xs, S_comp, ls="--", label="S component")
    ax.scatter(1e3 * xs, T_no_comp, ls="--", label="T (no decay)")
    ax.scatter(1e3 * xs, T_dec_comp, ls="--", label="T (decay)")

    # Annotate weights if available
    if 'weights' in ds:
        weights = fit.weights.values
        w_S, w_T_no, w_T_dec = weights[0], weights[1], weights[2]
        ax.text(0.02, 0.98, f"w_S={w_S:.2f}  w_T(no)={w_T_no:.2f}  w_T(dec)={w_T_dec:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)

    # Plot threshold line
    ax.axvline(
        1e3 * fit.I_threshold,
        color="r",
        linestyle="--",
        lw=1.5,
        label="Optimal threshold",
    )

    ax.set_xlabel("I Rotated [mV]")
    ax.set_ylabel("Density")
    ax.set_title(qubit_pair["qubit_pair"])
    ax.legend(fontsize=8)


def plot_confusion_matrices(ds: xr.Dataset, qubit_pairs: List[Any], fits: xr.Dataset):
    """
    Plots the confusion matrix for the given quantum dot pairs.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pairs : list
        A list of quantum dot pairs to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each quantum dot pair.
    - Each subplot contains the raw data and the fitted curve.
    """
    # Calculate grid dimensions based on number of qubit pairs
    n_pairs = len(qubit_pairs)
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))

    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols)

    # Flatten axes array for easy iteration
    if n_pairs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each qubit pair
    for idx, qubit_pair in enumerate(qubit_pairs):
        ax = axes[idx]
        qubit_pair_dict = {"qubit_pair": qubit_pair.id}
        plot_individual_confusion_matrix(ax, ds, qubit_pair_dict, fits.sel(qubit_pair=qubit_pair.id))

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Singlet and Triplet fidelity")
    fig.tight_layout()
    return fig


def plot_individual_confusion_matrix(ax: Axes, ds: xr.Dataset, qubit_pair: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual quantum dot pair data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit_pair : dict[str, str]
        mapping to the quantum dot pair to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    confusion = np.array([[float(fit.gg), float(fit.ge)], [float(fit.eg), float(fit.ee)]])
    ax.imshow(confusion)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels=["|S>", "|T>"])
    ax.set_yticklabels(labels=["|S>", "|T>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
    ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
    ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
    ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
    ax.set_title(qubit_pair["qubit_pair"])
