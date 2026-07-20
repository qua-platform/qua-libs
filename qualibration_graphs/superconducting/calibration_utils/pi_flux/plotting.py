"""Plotting utilities for pi vs flux calibration visualizations."""

from typing import Dict, List, Optional, Tuple, Union

from .analysis import FluxDistortionExpFitResult

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# ---------------------------------------------------------------------------
# Helper to get qubit names from either QubitList or plain list of strings
# ---------------------------------------------------------------------------


def _qubit_names(qubits) -> List[str]:
    """Return a list of qubit name strings regardless of input type."""
    # Abstracts away whether the caller passes a QubitList object or a plain list so all plotting functions can work uniformly.
    if hasattr(qubits, "get_names"):
        return qubits.get_names()
    return [q.name if hasattr(q, "name") else str(q) for q in qubits]


# ---------------------------------------------------------------------------
# Raw spectroscopy heatmaps
# ---------------------------------------------------------------------------


def plot_iq_abs_heatmap(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot |IQ| vs (time, frequency) as a pcolormesh for each qubit.

    Returns a single matplotlib Figure.
    """
    # Provides a 2-D overview of the raw spectroscopy response across all delay times, making it easy to visually verify that the qubit resonance dip is tracked correctly before fitting.
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        q_ds = ds.sel(qubit=qname)
        freq_ghz = q_ds["freq_full"].values / 1e9
        iq_abs = q_ds["IQ_abs"].values

        im = ax.pcolormesh(times, freq_ghz, iq_abs, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax).set_label("|IQ| (V)")
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(qname)

    scale = " [log]" if log_scale else ""
    fig.suptitle(f"|IQ| vs (time, freq){scale}", y=1.02)
    fig.tight_layout()
    return fig


def plot_phase_heatmap(ds: xr.Dataset, qubits):
    """Plot phase vs (time, frequency) for each qubit.

    Returns a single matplotlib Figure.
    """
    # Complements the |IQ| heatmap by showing the phase signature of the resonance, which can reveal phase-wrapping or calibration issues that are invisible in the amplitude channel.
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        q_ds = ds.sel(qubit=qname)
        freq_ghz = q_ds["freq_full"].values / 1e9
        phase = q_ds["phase"].values  # (time, detuning)

        im = ax.pcolormesh(
            times,
            freq_ghz,
            phase.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-np.pi,
            vmax=np.pi,
        )
        fig.colorbar(im, ax=ax).set_label("Phase (rad)")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(qname)

    fig.suptitle("Phase vs (time, freq)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1-D line plots
# ---------------------------------------------------------------------------


def plot_center_freqs(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot qubit frequency vs time for each qubit.

    When qubit objects expose ``xy.RF_frequency``, the y-axis shows the
    absolute qubit frequency (GHz).  Otherwise falls back to raw
    ``center_freqs`` values.

    Returns a single matplotlib Figure.
    """
    # Shows how the extracted qubit resonance frequency relaxes after the flux pulse, which is the intermediate result used to compute the flux step response before exponential fitting.
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    times = ds.time.values

    for ax, qname, q in zip(axes[0], names, qubits):
        cf = ds.sel(qubit=qname).center_freqs.values
        rf = getattr(getattr(q, "xy", None), "RF_frequency", None)
        if rf is not None:
            y = (cf + rf) / 1e9
            ax.set_ylabel("Qubit frequency (GHz)")
        else:
            y = cf / 1e9
            ax.set_ylabel("Center frequency (GHz)")
        ax.plot(times, y, marker="o", ms=4, lw=1.2)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
        ax.set_xlabel("Time (ns)")
        ax.set_title(qname)

    scale = " (log scale)" if log_scale else ""
    fig.suptitle(f"Qubit frequency shift vs time after flux pulse{scale}")
    fig.tight_layout()
    return fig


def plot_flux_response(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot flux response vs time for each qubit.

    Returns a single matplotlib Figure.
    """
    # Displays the flux step response (frequency converted to flux units) that the exponential model is fitted to; confirms whether the distortion tails are visible and well-resolved.
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        fr = ds.sel(qubit=qname).flux_response.values
        ax.plot(times, fr, lw=1.5)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Flux response (V)")
        ax.set_title(qname)

    scale = " (log scale)" if log_scale else ""
    fig.suptitle(f"Flux response vs time after flux pulse{scale}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectroscopy curve used for freq-to-flux mapping
# ---------------------------------------------------------------------------


def plot_spectroscopy_curve(ds: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Plot the extracted spectroscopy freq-vs-flux curve stored in *ds*.

    Reads from ``spec_curve_flux`` / ``spec_curve_freq`` data variables
    written by ``fit_raw_data()``.
    Returns a Figure, or ``None`` if no spectroscopy curve data is present.
    """
    # Lets the operator verify that the DP-extracted dispersion curve used for frequency-to-flux mapping is physically sensible before trusting the fitted distortion parameters.
    if "spec_curve_flux" not in ds or "spec_curve_freq" not in ds:
        return None

    run_id = ds.attrs.get("spectroscopy_run_id", "?")
    names = _qubit_names(qubits)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    spec_qubits = ds["spec_curve_flux"].spec_qubit.values.tolist()

    for ax, qname in zip(axes[0], names):
        if qname not in spec_qubits:
            ax.set_title(f"{qname} — no curve")
            continue
        flux_arr = ds["spec_curve_flux"].sel(spec_qubit=qname).values
        freq_arr = ds["spec_curve_freq"].sel(spec_qubit=qname).values / 1e9
        ax.plot(flux_arr, freq_arr, lw=1.5)
        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(qname)
        ax.grid(True)

    fig.suptitle(f"Spectroscopy curve used (run #{run_id})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exponential fit overlay (existing)
# ---------------------------------------------------------------------------


def plot_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict[str, FluxDistortionExpFitResult]):
    """
    Plots pi vs flux response with exponential decay fits for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the flux response data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fit_results : Dict
        The dictionary containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates plots for each qubit showing flux response over time.
    - Each plot contains the raw data and the fitted exponential curves.
    """
    # Iterates over qubits and delegates to plot_individual_fit to render the exponential model overlay, skipping qubits whose flux_response is entirely NaN.
    # grid = QubitGrid(ds, [q.grid_location for q in qubits])
    fig = None
    for q in qubits:
        t_data = ds.time.values
        y_data = ds.flux_response.sel(qubit=q.name).values
        if np.all(np.isnan(y_data)):
            continue

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
    # Renders the fitted sum-of-exponentials on both linear and log time axes so both fast and slow distortion poles are clearly visible for quality assessment.
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
    axs[1].legend(loc="best")
    axs[1].grid(True)

    fig.tight_layout()

    return fig, axs
