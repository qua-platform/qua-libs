"""Plotting utilities for pi vs flux calibration visualizations."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter


def _unpack_fit(q_fit, y_data):
    """Return ``(a_tau_tuple, a_dc)`` from a dataclass or dict fit result."""
    if hasattr(q_fit, "a_tau_tuple"):
        components = q_fit.a_tau_tuple if q_fit.a_tau_tuple is not None else []
        a_dc = getattr(q_fit, "a_dc", np.nan)
    elif isinstance(q_fit, dict):
        components = q_fit.get("a_tau_tuple") or q_fit.get("components") or []
        a_dc = q_fit.get("a_dc", np.nan)
    else:
        components, a_dc = [], np.nan
    if not components and np.all(np.isnan(y_data)):
        return [], np.nan
    return list(components), float(a_dc) if a_dc is not None else np.nan


def _exp_fit_curve(t_data, components, a_dc):
    """Return ``(y_fit, fit_text)`` for an exponential sum."""
    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"
    return y_fit, fit_text


# ---------------------------------------------------------------------------
# Per-axis plotters (qubit-agnostic; used by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def plot_center_freq(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    *,
    rf_frequency: Optional[float] = None,
    log_scale: bool = False,
) -> None:
    """Plot qubit frequency vs flux-pulse duration on one axis."""
    qname = qubit["qubit"]
    times = ds.time.values
    cf = ds.sel(qubit=qname).center_freqs.values
    if rf_frequency is not None:
        y = (cf + rf_frequency) / 1e9
        ylabel = "Qubit frequency (GHz)"
    else:
        y = cf / 1e9
        ylabel = "Center frequency (GHz)"
    ax.plot(times, y, marker="o", ms=4, lw=1.2)
    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, which="both")
    else:
        ax.grid(True)
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)


def plot_flux_response(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit: Any = None,
    *,
    log_scale: bool = False,
) -> None:
    """Plot flux step response vs time on one axis, optionally with the IIR fit."""
    qname = qubit["qubit"]
    t_data = np.asarray(ds.time.values, dtype=float)
    y_data = np.asarray(ds.flux_response.sel(qubit=qname).values, dtype=float)
    ax.plot(t_data, y_data, ".--", label="Data")

    if fit is not None:
        components, a_dc = _unpack_fit(fit, y_data)
        if components and np.isfinite(a_dc):
            y_fit, fit_text = _exp_fit_curve(t_data, components, a_dc)
            ax.plot(t_data, y_fit, "-", label="Fit")
            ax.text(
                0.98,
                0.5,
                fit_text,
                transform=ax.transAxes,
                fontsize=12,
                horizontalalignment="right",
                verticalalignment="center",
            )

    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, which="both")
    else:
        ax.grid(True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Flux response (V)", fontsize=14)
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=12)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict,
    *,
    debug: bool = False,
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default figures: flux response (with IIR fit) only.

    With ``debug=True``: spectroscopy heatmap (or center frequency trace),
    and the measured freq-vs-flux curve used for inversion when available.
    """
    grid_locations = [q.grid_location for q in qubits]
    rf_by_name = {
        q.name: getattr(getattr(q, "xy", None), "RF_frequency", None) for q in qubits
    }

    figures: Dict[str, plt.Figure] = {}

    grid_flux = QubitGrid(ds, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        plot_flux_response(
            ax,
            ds,
            qubit,
            fit=fit_results.get(qubit["qubit"]),
            log_scale=log_scale,
        )
    grid_flux.fig.suptitle("Flux response vs time after flux pulse", fontsize=16)
    grid_flux.fig.set_size_inches(15, 9)
    grid_flux.fig.tight_layout()
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    has_spectroscopy_map = "IQ_abs" in ds.data_vars or "state" in ds.data_vars
    if has_spectroscopy_map and "center_freqs" in ds:
        title_label = "|IQ|" if "IQ_abs" in ds.data_vars else "State"
        grid_spec = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_spec):
            plot_iq_abs(
                ax,
                ds,
                qubit,
                grid_spec.fig,
                log_scale=log_scale,
                overlay_center_freq=True,
                rf_frequency=rf_by_name.get(qubit["qubit"]),
            )
        grid_spec.fig.suptitle(
            f"Debug: {title_label} vs (time, frequency) with extracted center frequency",
            fontsize=16,
        )
        grid_spec.fig.set_size_inches(15, 9)
        grid_spec.fig.tight_layout()
        figures["spectroscopy"] = grid_spec.fig
    elif "center_freqs" in ds:
        grid_cf = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_cf):
            plot_center_freq(
                ax,
                ds,
                qubit,
                rf_frequency=rf_by_name.get(qubit["qubit"]),
                log_scale=log_scale,
            )
        grid_cf.fig.suptitle("Debug: Qubit frequency shift vs time after flux pulse", fontsize=16)
        grid_cf.fig.set_size_inches(15, 9)
        grid_cf.fig.tight_layout()
        figures["center_freq"] = grid_cf.fig

    if "spec_curve_flux" in ds and "spec_curve_freq" in ds:
        grid_curve = QubitGrid(ds, grid_locations)
        source_label = ds.attrs.get("freq_to_flux_sources", ds.attrs.get("freq_to_flux_source", "measured"))
        spec_qubits = ds["spec_curve_flux"].spec_qubit.values.tolist()
        n_plotted = 0
        for ax, qubit in grid_iter(grid_curve):
            qname = qubit["qubit"]
            plot_freq_vs_flux_curve(ax, ds, qubit, source_label=source_label)
            if qname in spec_qubits:
                flux_arr = ds["spec_curve_flux"].sel(spec_qubit=qname).values
                freq_arr = ds["spec_curve_freq"].sel(spec_qubit=qname).values
                if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                    n_plotted += 1
        if n_plotted:
            grid_curve.fig.suptitle(f"Debug: Freq-vs-flux curve used ({source_label})", fontsize=16)
            grid_curve.fig.set_size_inches(15, 9)
            grid_curve.fig.tight_layout()
            figures["freq_vs_flux_curve"] = grid_curve.fig
        else:
            plt.close(grid_curve.fig)

    return figures


# ---------------------------------------------------------------------------
# Debug figures (``debug_plots=True``)
# ---------------------------------------------------------------------------


def plot_iq_abs(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fig: plt.Figure,
    *,
    log_scale: bool = False,
    overlay_center_freq: bool = False,
    rf_frequency: Optional[float] = None,
) -> None:
    """Plot |IQ| or state heatmap vs (time, frequency) on one axis."""
    if "IQ_abs" in ds.data_vars:
        signal_var = "IQ_abs"
        cbar_label = "|IQ| (V)"
    else:
        signal_var = "state"
        cbar_label = "State"

    qname = qubit["qubit"]
    times = ds.time.values
    q_ds = ds.sel(qubit=qname)
    freq_ghz = q_ds["freq_full"].values / 1e9
    freq_dim = "detuning" if "detuning" in q_ds[signal_var].dims else "freq"
    signal = q_ds[signal_var].transpose(freq_dim, "time").values
    im = ax.pcolormesh(times, freq_ghz, signal, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax).set_label(cbar_label)

    if overlay_center_freq and "center_freqs" in ds:
        cf = np.asarray(q_ds.center_freqs.values, dtype=float)
        if rf_frequency is not None:
            freq_line_ghz = (cf + rf_frequency) / 1e9
        else:
            freq_line_ghz = cf / 1e9
        ax.plot(
            times,
            freq_line_ghz,
            color="darkred",
            lw=2.5,
            marker="o",
            ms=7,
            mfc="darkred",
            mec="white",
            mew=1.0,
            label="Center freq",
            zorder=5,
        )
        ax.legend(loc="upper right", fontsize=10)

    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, which="both")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)


def plot_freq_vs_flux_curve(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    *,
    source_label: str = "measured",
) -> None:
    """Plot the freq-vs-flux curve used for inversion on one axis."""
    qname = qubit["qubit"]
    if "spec_curve_flux" not in ds or "spec_curve_freq" not in ds:
        ax.set_title(f"{qname} — no curve")
        return
    spec_qubits = ds["spec_curve_flux"].spec_qubit.values.tolist()
    if qname not in spec_qubits:
        ax.set_title(f"{qname} — no curve")
        return
    flux_arr = ds["spec_curve_flux"].sel(spec_qubit=qname).values
    freq_arr = ds["spec_curve_freq"].sel(spec_qubit=qname).values / 1e9
    if not np.isfinite(flux_arr).any() or not np.isfinite(freq_arr).any():
        ax.set_title(f"{qname} — no curve")
        return
    ax.plot(flux_arr, freq_arr, lw=1.5)
    ax.set_xlabel("Flux bias (V)")
    ax.set_ylabel("Qubit frequency (GHz)")
    ax.set_title(qname)
    ax.grid(True)


def plot_spectroscopy_curve(ds: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Plot measured freq-vs-flux curve(s) on a single multi-qubit figure (legacy helper)."""
    if "spec_curve_flux" not in ds or "spec_curve_freq" not in ds:
        return None

    source_label = ds.attrs.get("freq_to_flux_sources", ds.attrs.get("freq_to_flux_source", "measured"))
    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    spec_qubits = ds["spec_curve_flux"].spec_qubit.values.tolist()
    n_plotted = 0
    for ax, qname in zip(axes[0], names):
        plot_freq_vs_flux_curve(ax, ds, {"qubit": qname}, source_label=source_label)
        if qname in spec_qubits:
            flux_arr = ds["spec_curve_flux"].sel(spec_qubit=qname).values
            freq_arr = ds["spec_curve_freq"].sel(spec_qubit=qname).values
            if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                n_plotted += 1
    if not n_plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"Debug: Freq-vs-flux curve used ({source_label})", fontsize=16)
    fig.tight_layout()
    return fig
