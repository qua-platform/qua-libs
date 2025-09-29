from typing import Dict, List, Tuple

import numpy as np
from quam import components
import xarray as xr
import matplotlib.pyplot as plt
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

def plot_new_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict):
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

        components = fit_results[q.name].best_components
        a_dc = fit_results[q.name].best_a_dc

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

def plot_pi_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict[str, object] | None = None):
    figures: Dict[str, plt.Figure] = {}

    # Raw
    if ("state" in ds.data_vars and ("freq_full" in ds.coords or "freq" in ds.coords)) or "IQ_abs" in ds.data_vars or "I" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            if "state" in ds.data_vars and "freq_full" in ds.coords:
                im = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
                    ax=ax, add_colorbar=False, x="time", y="freq_GHz"
                )
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit State")
            elif "state" in ds.data_vars and "freq" in ds.coords:
                im = ds.assign_coords(freq_GHz=ds.freq / 1e9).loc[qubit].state.plot(
                    ax=ax, add_colorbar=False, x="time", y="freq_GHz"
                )
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit State")
            elif "IQ_abs" in ds.data_vars:
                im = ds.loc[qubit].IQ_abs.plot(ax=ax, add_colorbar=False, x="time", y="detuning")
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Amplitude (V)")
            elif "I" in ds.data_vars:
                im = ds.loc[qubit].I.plot(ax=ax, add_colorbar=False, x="time", y="detuning")
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("I Quadrature (V)")
        grid.fig.suptitle("Qubit spectroscopy vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_raw"] = grid.fig

    # Center frequencies
    if "center_freqs" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            cf = ds.loc[qubit].center_freqs
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(cf.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y / 1e9)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
        grid.fig.suptitle("Qubit frequency shift vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_freqs_shift"] = grid.fig

        # Log-scale version
        grid_log = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_log):
            cf = ds.loc[qubit].center_freqs
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(cf.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y / 1e9)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.set_xscale('log')
            ax.grid(True)
        grid_log.fig.suptitle("Qubit frequency shift vs time after flux pulse")
        grid_log.fig.set_size_inches(15, 9)
        grid_log.fig.tight_layout()
        figures["figure_freqs_shift_log"] = grid_log.fig

    # Flux and fits
    if "flux_response" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            qb_flux = ds.loc[qubit].flux_response
            try:
                final_val = float(qb_flux.isel(time=-1).values)
            except Exception:
                final_val = np.nan
            if np.isfinite(final_val) and abs(final_val) > 1e-12:
                qb_flux_plot = qb_flux / final_val
            else:
                max_abs = float(np.nanmax(np.abs(qb_flux.values)))
                qb_flux_plot = qb_flux / max_abs if max_abs > 0 else qb_flux
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(qb_flux_plot.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y)
            if fit_results and fit_results.get(qubit["qubit"], None) is not None:
                res = fit_results[qubit["qubit"]]
                # Support both dict-like and dataclass result
                best_a_dc = res["best_a_dc"] if isinstance(res, dict) else res.best_a_dc
                components = res["best_components"] if isinstance(res, dict) else res.best_components
                t_data = ds.loc[qubit].time.values
                y_fit = np.ones_like(t_data) * best_a_dc
                for amp, tau in components:
                    y_fit += amp * np.exp(-(t_data - t_data[0]) / tau)
                if np.isfinite(final_val) and abs(final_val) > 1e-12:
                    y_fit = y_fit / final_val
                else:
                    if max_abs > 0:
                        y_fit = y_fit / max_abs
                ax.plot(t_data, y_fit, "r-", linewidth=2)
            ax.set_ylabel("Flux (normalized)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.grid(True)
        grid.fig.suptitle("Flux response vs time")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_flux_response"] = grid.fig

        # Log-scale version
        grid_log = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_log):
            qb_flux = ds.loc[qubit].flux_response
            try:
                final_val = float(qb_flux.isel(time=-1).values)
            except Exception:
                final_val = np.nan
            if np.isfinite(final_val) and abs(final_val) > 1e-12:
                qb_flux_plot = qb_flux / final_val
            else:
                max_abs = float(np.nanmax(np.abs(qb_flux.values)))
                qb_flux_plot = qb_flux / max_abs if max_abs > 0 else qb_flux
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(qb_flux_plot.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y)
            ax.set_ylabel("Flux (normalized)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.set_xscale('log')
            ax.grid(True)
        grid_log.fig.suptitle("Flux response vs time")
        grid_log.fig.set_size_inches(15, 9)
        grid_log.fig.tight_layout()
        figures["figure_flux_response_log"] = grid_log.fig

    return figures

