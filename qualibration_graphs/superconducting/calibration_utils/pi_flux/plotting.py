from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict):
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
        y_data = ds.flux_response.sel(qubit=q.name).values

        components = fit_results[q.name]["a_tau_tuple"]
        a_dc = fit_results[q.name]["a_dc"]

        # Guard against NaN or None DC term for formatting & model building
        if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
            # If we can't determine DC term, approximate from tail of data
            a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

        fig, _ = plot_individual_fit(t_data, y_data, components=components, a_dc=a_dc)

    return fig


def plot_individual_fit(t_data: np.ndarray, y_data: np.ndarray, components: List[Tuple[float, float]], a_dc: float):
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
