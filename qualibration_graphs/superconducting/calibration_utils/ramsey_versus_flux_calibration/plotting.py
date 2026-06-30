"""Plotting functions for Ramsey versus flux calibration."""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.analysis import lorentzian_peak
from qualibration_libs.plotting import QubitGrid, grid_iter

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots Ramsey vs flux with the fitted dataset.

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

    grid.fig.suptitle("Ramsey vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_parabolas_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots Ramsey vs flux frequency with the fitted dataset.

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
        plot_individual_parabolas_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Ramsey vs flux frequency")
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

    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle_time (uS)")
    ax.set_ylabel(" Flux (V)")

    flux_offset = fit.flux_offset

    ax.axhline(flux_offset, color="red", linestyle="--", label="Flux offset")


def plot_individual_parabolas_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    detuning = fit.artifitial_detuning

    (fit.unfolded_frequency * 1e3 - detuning).plot(ax=ax, linestyle="--", marker=".", label="Unfolded freq")

    # Show aliased (raw) fitted frequency for comparison
    (np.abs(fit.sel(fit_vals="f").fit_results) * 1e3 - detuning).plot(
        ax=ax, linestyle=":", marker=".", alpha=0.35, color="gray", label="Aliased freq"
    )

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux offset (V)")
    ax.set_ylabel("Qubit SweetSpot detuning (MHz)")

    # Parabola fit: f(V) = c2*(V - V0)² + f0, converted to MHz minus detuning
    flux = fit.unfolded_frequency.flux_bias.values
    c2_mhz = -float(fit.quad_term) / 1e3  # MHz/V²
    v0 = float(fit.flux_offset)
    f0_mhz = float(fit.freq_offset) * 1e-3  # MHz at vertex
    parabola_mhz = c2_mhz * (flux - v0) ** 2 + f0_mhz - float(detuning)
    ax.plot(flux, parabola_mhz, color="tab:orange", linewidth=2, alpha=0.8, label="Parabola fit")

    flux_offset = fit.flux_offset
    ax.axvline(flux_offset, color="red", linestyle="--", label="Flux offset")

    freq_offset = fit.freq_offset.values * 1e-3 - detuning
    ax.axhline(freq_offset, color="green", linestyle="--", label="Detuning from SweetSpot")

    # Nyquist zone boundary lines
    max_zone = int(fit.nyquist_zones.max())
    f_nyq = float(fit.f_nyquist)
    for n in range(1, max_zone + 1):
        boundary_mhz = n * f_nyq * 1e3 - detuning
        ax.axhline(
            boundary_mhz,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label="Nyquist boundary" if n == 1 else None,
        )

    ax.legend(fontsize="small")


def plot_qubit_freq_vs_flux_fig(ds: xr.Dataset, qubits) -> "Figure":
    """Plot absolute qubit RF frequency vs flux for all qubits.

    The curve is derived from the unfolded Ramsey frequency and the corrected
    RF_frequency stored in ds["f_qubit_vs_flux"] (GHz).  Qubits whose
    fitting failed have an all-NaN row and are labelled accordingly.

    Parameters
    ----------
    ds : xr.Dataset
        Fit dataset containing f_qubit_vs_flux (GHz), flux_offset,
        freq_offset, and quad_term.
    qubits : list of AnyTransmon
        Active qubits (used for grid layout).

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_qubit_freq_vs_flux(ax, ds, qubit)

    grid.fig.suptitle("Qubit RF frequency vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_qubit_freq_vs_flux(ax, ds: xr.Dataset, qubit: dict) -> None:
    """Plot absolute qubit RF frequency vs flux on a single axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    ds : xr.Dataset
        Fit dataset (must contain f_qubit_vs_flux in GHz).
    qubit : dict
        Mapping with key "qubit" giving the qubit name.
    """
    import numpy as np

    q_name = qubit["qubit"]
    f_ghz = ds["f_qubit_vs_flux"].sel(qubit=q_name).values
    flux = ds.flux_bias.values

    if np.all(np.isnan(f_ghz)):
        ax.set_title(q_name)
        ax.text(
            0.5,
            0.5,
            "fitting failed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="red",
        )
        ax.set_xlabel("Flux offset (V)")
        ax.set_ylabel("RF frequency (GHz)")
        return

    ax.plot(flux, f_ghz, linestyle="-", marker=".", label="f_qubit (Ramsey)")

    # Parabola fit overlay (same curve, smooth)
    c2_ghz = -float(ds.quad_term.sel(qubit=q_name).values) / 1e9  # Hz/V2 -> GHz/V2... actually quad_term unit check
    # quad_term stored as MHz/V^2 (= -1e6 * GHz/V^2 polyfit coeff * 1e3... see analysis)
    # safe approach: just use the fitted f_qubit_vs_flux data itself for the parabola
    # Fit a smooth parabola through the valid (non-NaN) points for visual overlay
    valid = ~np.isnan(f_ghz)
    if valid.sum() >= 3:
        coeffs = np.polyfit(flux[valid], f_ghz[valid], 2)
        flux_smooth = np.linspace(flux[valid].min(), flux[valid].max(), 300)
        ax.plot(
            flux_smooth,
            np.polyval(coeffs, flux_smooth),
            color="tab:orange",
            linewidth=2,
            alpha=0.8,
            label="Parabola fit",
        )

    # Sweet-spot vertical line
    flux_offset = float(ds.flux_offset.sel(qubit=q_name).values)
    ax.axvline(flux_offset, color="red", linestyle="--", label="Flux offset")

    # Operating frequency horizontal line (value at flux_offset)
    f_at_sweetspot = float(np.interp(flux_offset, flux[valid], f_ghz[valid])) if valid.sum() > 0 else np.nan
    if not np.isnan(f_at_sweetspot):
        ax.axhline(f_at_sweetspot, color="green", linestyle="--", label=f"Sweet-spot freq ({f_at_sweetspot:.4f} GHz)")

    ax.set_title(q_name)
    ax.set_xlabel("Flux offset (V)")
    ax.set_ylabel("RF frequency (GHz)")
    ax.legend(fontsize="small")
