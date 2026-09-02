"""Plotting utilities for pi vs flux calibration visualizations."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon


# ---------------------------------------------------------------------------
# Default figures (always produced by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def plot_center_freqs(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot qubit frequency vs flux-pulse duration for each qubit.

    When qubit objects expose ``xy.RF_frequency``, the y-axis shows absolute
    qubit frequency (GHz); otherwise raw ``center_freqs``.
    """
    names = [q.name for q in qubits]
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
    """Plot flux step response vs time for each qubit."""
    names = [q.name for q in qubits]
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


def plot_individual_raw_data_with_fit(ax_lin, ax_log, t_data, y_data, components, a_dc):
    """Draw sum-of-exponentials data+fit on a linear and a log axis (one qubit)."""
    fit_text = f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"

    ax_lin.plot(t_data, y_data, ".--", label="Data")
    ax_lin.plot(t_data, y_fit, label="Fit")
    ax_lin.text(
        0.98, 0.5, fit_text, transform=ax_lin.transAxes, fontsize=10,
        horizontalalignment="right", verticalalignment="center",
    )
    ax_lin.set_xlabel("Time (ns)")
    ax_lin.set_ylabel("Flux Response")
    ax_lin.legend()
    ax_lin.grid(True)
    ax_lin.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax_log.plot(t_data, y_data, ".--", label="Data")
    ax_log.plot(t_data, y_fit, label="Fit")
    ax_log.text(
        0.98, 0.5, fit_text, transform=ax_log.transAxes, fontsize=10,
        horizontalalignment="right", verticalalignment="center",
    )
    ax_log.set_xlabel("Time (ns)")
    ax_log.set_ylabel("Flux Response")
    ax_log.set_xscale("log")
    ax_log.legend(loc="best")
    ax_log.grid(True)


def plot_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict):
    """Flux response + multi-exp fit (one linear/log pair per qubit)."""
    names = [q.name for q in qubits]
    n = len(names)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n), squeeze=False)
    t_data = ds.time.values

    for row, qname in enumerate(names):
        ax_lin, ax_log = axes[row]
        y_data = ds.flux_response.sel(qubit=qname).values
        if np.all(np.isnan(y_data)):
            for ax in (ax_lin, ax_log):
                ax.set_title(f"{qname} — no data")
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("Flux Response")
            continue

        components = fit_results[qname]["a_tau_tuple"]
        a_dc = fit_results[qname]["a_dc"]
        if a_dc is None or (isinstance(a_dc, (float, np.floating)) and np.isnan(a_dc)):
            a_dc = float(y_data[-5:].mean()) if len(y_data) >= 5 else float(y_data.mean())

        plot_individual_raw_data_with_fit(
            ax_lin, ax_log, t_data, y_data, components=components, a_dc=a_dc
        )
        ax_lin.set_title(qname)
        ax_log.set_title(f"{qname} (log)")

    fig.tight_layout()
    return fig


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict,
    *,
    debug: bool = False,
    ramsey_run_id: Optional[int] = None,
) -> Dict[str, plt.Figure]:
    """Default 17a figures: center freq, flux response, and exponential fit.

    Parameters
    ----------
    debug :
        If True, also add IQ/phase heatmaps and spectroscopy/Ramsey reference
        curves when available.
    ramsey_run_id :
        Optional override for the Ramsey reference plot (only used when
        ``debug=True``).

    Returns
    -------
    dict
        Always: ``center_freq_{linear,log}``, ``flux_response_{linear,log}``,
        ``fitted_data``. With ``debug=True``, may also include ``iq_abs_*``,
        ``phase``, ``spectroscopy_curve``, ``ramsey_curve``.
    """
    figures: Dict[str, plt.Figure] = {
        "center_freq_linear": plot_center_freqs(ds, qubits, log_scale=False),
        "center_freq_log": plot_center_freqs(ds, qubits, log_scale=True),
        "flux_response_linear": plot_flux_response(ds, qubits, log_scale=False),
        "flux_response_log": plot_flux_response(ds, qubits, log_scale=True),
        "fitted_data": plot_fit(ds, qubits, fit_results),
    }

    if not debug:
        return figures

    for key, fig in {
        "iq_abs_linear": plot_iq_abs_heatmap(ds, qubits, log_scale=False),
        "iq_abs_log": plot_iq_abs_heatmap(ds, qubits, log_scale=True),
        "phase": plot_phase_heatmap(ds, qubits),
        "spectroscopy_curve": plot_spectroscopy_curve(ds, qubits),
        "ramsey_curve": plot_ramsey_curve(qubits, ramsey_run_id),
    }.items():
        if fig is not None:
            figures[key] = fig

    return figures


# ---------------------------------------------------------------------------
# Debug figures (``debug_plots=True``)
# ---------------------------------------------------------------------------


def plot_iq_abs_heatmap(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Raw |IQ| (or state) vs (time, frequency) heatmap per qubit."""
    if "IQ_abs" in ds.data_vars:
        signal_var = "IQ_abs"
        cbar_label = "|IQ| (V)"
        title_label = "|IQ|"
    elif "state" in ds.data_vars:
        signal_var = "state"
        cbar_label = "State"
        title_label = "State"
    else:
        return None

    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        q_ds = ds.sel(qubit=qname)
        freq_ghz = q_ds["freq_full"].values / 1e9
        _fd = "detuning" if "detuning" in q_ds[signal_var].dims else "freq"
        signal = q_ds[signal_var].transpose(_fd, "time").values

        im = ax.pcolormesh(times, freq_ghz, signal, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax).set_label(cbar_label)
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(qname)

    scale = " [log]" if log_scale else ""
    fig.suptitle(f"{title_label} vs (time, freq){scale}", y=1.02)
    fig.tight_layout()
    return fig


def plot_phase_heatmap(ds: xr.Dataset, qubits):
    """Phase vs (time, frequency) heatmap per qubit (IQ acquisition only)."""
    if "phase" not in ds.data_vars:
        return None

    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        q_ds = ds.sel(qubit=qname)
        freq_ghz = q_ds["freq_full"].values / 1e9
        _fd = "detuning" if "detuning" in q_ds["phase"].dims else "freq"
        phase = q_ds["phase"].transpose(_fd, "time").values

        im = ax.pcolormesh(
            times, freq_ghz, phase,
            shading="auto", cmap="RdBu_r", vmin=-np.pi, vmax=np.pi,
        )
        fig.colorbar(im, ax=ax).set_label("Phase (rad)")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(qname)

    fig.suptitle("Phase vs (time, freq)", y=1.02)
    fig.tight_layout()
    return fig


def plot_spectroscopy_curve(ds: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Spectroscopy freq-vs-flux curve attached to ``ds`` during analysis."""
    if "spec_curve_flux" not in ds or "spec_curve_freq" not in ds:
        return None

    run_id = ds.attrs.get("spectroscopy_run_id", "?")
    names = [q.name for q in qubits]
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


def plot_ramsey_curve(qubits, run_id: Optional[int] = None) -> Optional[plt.Figure]:
    """Ramsey vs Z-flux reference (reloaded from run id / extras)."""
    from calibration_utils.common_utils.flux_distortions.curves import load_ramsey_curve

    n = len(qubits)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    n_loaded = 0
    for ax, qubit in zip(axes[0], qubits):
        curve = load_ramsey_curve(qubit, run_id)
        if curve is None:
            continue
        n_loaded += 1
        flux_bias, qubit_freq = curve
        ax.plot(flux_bias, np.array(qubit_freq) / 1e9, marker=".", linestyle="-")
        ax.set_xlabel("Z flux (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(qubit.name)
    fig.suptitle(
        "Ramsey vs Z-flux (param override or extras load id)"
        if n_loaded
        else "Ramsey vs Z-flux — no run IDs found"
    )
    fig.tight_layout()
    return fig
