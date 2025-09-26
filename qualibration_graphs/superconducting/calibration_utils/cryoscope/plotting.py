from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cryoscope import expdecay, two_expdecay
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_raw_data(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Raw Cryoscope Data")
    grid.fig.set_size_inches(5, 5)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis.

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

    if hasattr(fit, "I"):
        data = "I"
        label = "Rotated I quadrature [mV]"
    elif hasattr(fit, "state"):
        data = "state"
        label = "Qubit state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    # ds[data].sel(qubit=qubit["qubit"]).plot.line(ax=ax, x="time", linestyle="--", marker=".")

    ax.scatter(fit[data].sel(axis="x"), fit[data].sel(axis="y"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"qubit = {qubit['qubit']}")


def plot_normalized_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_flux(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Reconstructed normalized Flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_new_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for q in qubits:
        t_data = ds.time.values
        y_data = ds.flux.sel(qubit=q.name).values

        # Access the variable containing fit attrs
        fit_var = fits.fit_results
        attrs = getattr(fit_var, "attrs", {})

        # Reconstruct components from new 1D attribute style (preferred)
        amps = attrs.get("fit_component_amps")
        taus = attrs.get("fit_component_taus_ns")
        components = []
        if amps is not None and taus is not None:
            try:
                components = list(zip(list(amps), list(taus)))
            except Exception:
                components = []
        else:
            # Backward compatibility: older (in-memory) attribute containing list of tuples
            legacy = attrs.get("fit_components")
            if isinstance(legacy, (list, tuple)):
                components = legacy

        a_dc = attrs.get("fit_a_dc", np.nan)
        # Guard against NaN or None DC term for formatting & model building
        if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
            # If we can't determine DC term, approximate from tail of data
            a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

        fig, axs = plot_individual_new_fit(t_data, y_data, components=components, a_dc=a_dc)

    # grid.fig.suptitle("Reconstructed normalized Flux")
    # grid.fig.set_size_inches(15, 9)
    # grid.fig.tight_layout()
    return fig


def plot_individual_new_fit(t_data: np.ndarray, y_data: np.ndarray, components: List[Tuple[float, float]], a_dc: float):
    """Plot exponential fit results with both linear and log scales.

    Args:
        t_data (np.ndarray): Time points in nanoseconds
        y_data (np.ndarray): Measured flux response data
        components (List[Tuple[float, float]]): List of (amplitude, tau) pairs for each fitted component
        a_dc (float): Constant term

    Returns:
        tuple: (fig, axs) where:
            - fig: Figure object
            - axs: List of axes objects
    """

    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot - linear scale
    axs[0].plot(t_data, y_data, ".--", label="Data")
    axs[0].plot(t_data, y_fit, label="Fit")
    axs[0].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[0].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[0].set_xlabel("Time (ns)")
    axs[0].set_ylabel("Flux Response")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Second subplot - log scale
    axs[1].plot(t_data, y_data, ".--", label="Data")
    axs[1].plot(t_data, y_fit, label="Fit")
    axs[1].text(
        0.98,
        0.5,
        fit_text,
        transform=axs[1].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
    )
    axs[1].set_xlabel("Time (ns)")
    axs[1].set_ylabel("Flux Response")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.tight_layout()

    return fig, axs


def plot_individual_flux(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis.

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
    - If the fit dataset is provided and successful, the fitted curve is plotted along with the raw data.
    - If the fit failed, only the raw data is plotted with appropriate annotations.
    """
    # Always plot the raw flux data
    fit.fit_results.flux.plot(ax=ax, linestyle="--", marker=".", label="Raw flux data")

    # Check if fits are available and successful
    fit1_success = fit.fit_results.attrs.get("fit_1exp_success", False)
    fit2_success = fit.fit_results.attrs.get("fit_2exp_success", False)

    if fit1_success:
        fit1_params = fit.fit_results.attrs.get("fit_1exp")
        if fit1_params is not None:
            ax.plot(fit.time, expdecay(fit.time, *fit1_params), label="Single exp fit", linewidth=3, alpha=0.8)

    if fit2_success:
        fit2_params = fit.fit_results.attrs.get("fit_2exp")
        if fit2_params is not None:
            ax.plot(fit.time, two_expdecay(fit.time, *fit2_params), label="Double exp fit", linewidth=3, alpha=0.8)

    # Add status annotation
    status_text = []
    if fit1_success:
        status_text.append("Single exp: ✓")
    else:
        status_text.append("Single exp: ✗")

    if fit2_success:
        status_text.append("Double exp: ✓")
    else:
        status_text.append("Double exp: ✗")

    ax.text(
        0.02,
        0.98,
        "\n".join(status_text),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    ax.legend()
    ax.set_title(f"Qubit {qubit['qubit']}")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Normalized flux")


def plot_raw_data_only(ds: xr.Dataset, qubits: List[AnyTransmon]):
    """
    Plots only the raw cryoscope data when fits are not available.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_only(ax, ds, qubit)

    grid.fig.suptitle("Raw Cryoscope Data (No fits available)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_only(ax: Axes, ds: xr.Dataset, qubit: dict[str, str]):
    """
    Plots individual raw qubit data when no fits are available.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    """
    try:
        if hasattr(ds, "I"):
            data = "I"
            label = "Rotated I quadrature [mV]"
        elif hasattr(ds, "state"):
            data = "state"
            label = "Qubit state"
        else:
            raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

        # Plot raw data vs time for each frame
        qubit_data = ds[data].sel(qubit=qubit["qubit"])

        # Plot a subset of frames to avoid cluttering
        num_frames = qubit_data.sizes.get("frame", 1)
        frame_step = max(1, num_frames // 5)  # Show max 5 frames

        for i, frame_idx in enumerate(range(0, num_frames, frame_step)):
            frame_data = qubit_data.isel(frame=frame_idx)
            alpha = 0.7 if num_frames > 1 else 1.0
            ax.plot(
                frame_data.time,
                frame_data.values,
                linestyle="--",
                marker=".",
                alpha=alpha,
                label=f"Frame {frame_idx}" if num_frames > 1 and i < 3 else "",
            )

        ax.set_title(f"Qubit {qubit['qubit']} - Fit Failed")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel(label)

        if num_frames > 1:
            ax.legend()

        ax.text(
            0.02,
            0.98,
            "Fits failed - showing raw data only",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

    except Exception as e:
        # Fallback if even raw data plotting fails
        ax.text(
            0.5,
            0.5,
            f"Error plotting data:\n{str(e)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
        )
        ax.set_title(f"Qubit {qubit['qubit']} - Error")
