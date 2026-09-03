"""Plotting utilities for cryoscope (17c) visualizations."""

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


def _flux_fit_to_freq(freq: np.ndarray, flux: np.ndarray, flux_fit: np.ndarray) -> np.ndarray:
    """Map a fitted flux trace back to frequency via the empirical quadratic scale.

    Uses ``freq ≈ c * sign(freq) * flux²`` with ``c`` estimated from the measured
    traces, so the IIR model (fit in flux) can be overlaid on the frequency plot.
    """
    flux = np.asarray(flux, dtype=float)
    freq = np.asarray(freq, dtype=float)
    flux_fit = np.asarray(flux_fit, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.nanmedian(np.abs(freq) / np.square(flux))
    if not np.isfinite(scale) or scale <= 0:
        return np.full_like(freq, np.nan)
    sign = np.sign(freq)
    sign[sign == 0] = 1.0
    return sign * scale * np.square(flux_fit)


# ---------------------------------------------------------------------------
# Per-axis plotters (qubit-agnostic; used by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def plot_cryoscope_freq(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit: Any = None,
    *,
    log_scale: bool = False,
) -> None:
    """Plot cryoscope frequency vs time on *ax*, optionally with the IIR model.

    The exponential fit lives in flux space; when *fit* is provided it is mapped
    back to frequency for the overlay.
    """
    qname = qubit["qubit"]
    t_data = np.asarray(ds.time.values, dtype=float)
    freq = np.asarray(ds["freq"].sel(qubit=qname).values, dtype=float)
    ax.plot(t_data, freq, ".--", label="Data")

    if fit is not None:
        flux = np.asarray(ds["flux_response"].sel(qubit=qname).values, dtype=float)
        components, a_dc = _unpack_fit(fit, flux)
        if components and np.isfinite(a_dc):
            flux_fit, fit_text = _exp_fit_curve(t_data, components, a_dc)
            freq_fit = _flux_fit_to_freq(freq, flux, flux_fit)
            ax.plot(t_data, freq_fit, "-", label="Fit")
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
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Cryoscope frequency (GHz)", fontsize=14)
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=12)


def plot_flux_response(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit: Any = None,
    *,
    log_scale: bool = False,
) -> None:
    """Plot flux step response vs time on *ax*, optionally with the IIR fit."""
    qname = qubit["qubit"]
    t_data = np.asarray(ds.time.values, dtype=float)
    flux = np.asarray(ds["flux_response"].sel(qubit=qname).values, dtype=float)
    ax.plot(t_data, flux, ".--", label="Data")

    if fit is not None:
        components, a_dc = _unpack_fit(fit, flux)
        if components and np.isfinite(a_dc):
            flux_fit, fit_text = _exp_fit_curve(t_data, components, a_dc)
            ax.plot(t_data, flux_fit, "-", label="Fit")
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
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Flux response (V)", fontsize=14)
    ax.set_title(qname)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=12)


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubits,
    fit_results: dict,
    *,
    debug: bool = False,
    fir_results: Optional[dict] = None,
) -> Dict[str, plt.Figure]:
    """Default 17c figures: cryoscope freq and flux response (both with IIR fit).

    Uses ``QubitGrid`` / ``grid_iter`` and calls the per-axis plotters once per
    qubit. When ``fir_results`` is set, also builds a compact FIR summary.

    Returns
    -------
    dict
        Always: ``cryoscope_freq``, ``flux_response``.
        With ``debug=True``: may also include ``unwrapped_phase``,
        ``freq_vs_flux_curve``. With FIR results: compact ``fir_*`` summary.
    """
    grid_locations = [q.grid_location for q in qubits]

    grid_freq = QubitGrid(ds_fit, grid_locations)
    for ax, qubit in grid_iter(grid_freq):
        plot_cryoscope_freq(ax, ds_fit, qubit, fit=fit_results.get(qubit["qubit"]))
    grid_freq.fig.suptitle("Cryoscope frequency vs time", fontsize=16)
    grid_freq.fig.set_size_inches(15, 9)
    grid_freq.fig.tight_layout()

    grid_flux = QubitGrid(ds_fit, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        plot_flux_response(ax, ds_fit, qubit, fit=fit_results.get(qubit["qubit"]))
    grid_flux.fig.suptitle("Flux response vs time", fontsize=16)
    grid_flux.fig.set_size_inches(15, 9)
    grid_flux.fig.tight_layout()

    figures: Dict[str, plt.Figure] = {
        "cryoscope_freq": grid_freq.fig,
        "flux_response": grid_flux.fig,
    }

    if debug:
        for key, fig in {
            "unwrapped_phase": plot_unwrapped_phase(ds_fit, qubits),
            "freq_vs_flux_curve": plot_spectroscopy_curve(ds_fit, qubits),
        }.items():
            if fig is not None:
                figures[key] = fig

    if fir_results:
        for q in qubits:
            qname = q.name
            res = fir_results.get(qname)
            if res is None or not res.get("success"):
                continue
            fig_fir, axes_fir = plt.subplots(1, 2, figsize=(12, 4))
            plot_fir(axes_fir, res)
            fig_fir.suptitle(f"FIR summary — {qname}")
            fig_fir.tight_layout()
            figures[f"fir_{qname}"] = fig_fir

    return figures


# ---------------------------------------------------------------------------
# FIR diagnostics (per-qubit; orchestrated by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def plot_fir(axes, fir_result: dict) -> None:
    """Draw FIR diagnostics for one qubit onto a 1×2 axes grid.

    Layout:: ``[ resampled 1↔2 GS/s | corrected response ]``

    *axes* may be a length-2 array from ``plt.subplots(1, 2)`` or a flat
    sequence ``(ax_resampled, ax_corrected)``.
    """
    flat = np.asarray(axes, dtype=object).ravel()
    if flat.size != 2:
        raise ValueError("plot_fir expects a 1×2 axes grid (2 axes)")
    ax_r, ax_c = flat

    t1 = np.asarray(fir_result["time_1gs"])
    t2 = np.asarray(fir_result["time_2gs"])

    # Resampled flux (check upsampling before the FIR fit)
    ax_r.plot(t1, fir_result["normalized_1gs"], "b.-", label="1 GS/s", alpha=0.6)
    ax_r.plot(t2, fir_result["normalized_2gs"], "r.-", ms=3, label="2 GS/s", alpha=0.6)
    ax_r.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_r.set_xlabel("Time (ns)")
    ax_r.set_ylabel("Normalized amplitude")
    ax_r.set_title("Resampled flux")
    ax_r.legend()
    ax_r.grid(True, alpha=0.3)

    # Corrected response (validate inverse FIR)
    ax_c.plot(t1, fir_result["normalized_1gs"], label="data")
    ax_c.plot(t1, fir_result["corrected_1gs"], "--", label="FIR-corrected")
    ax_c.axhline(1.001, color="k", lw=0.8, ls="--", label="±0.1%")
    ax_c.axhline(0.999, color="k", lw=0.8, ls="--")
    ax_c.set_xlabel("Time (ns)")
    ax_c.set_ylabel("Amplitude")
    ax_c.set_title("Corrected response")
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Debug figures (``debug_plots=True``)
# ---------------------------------------------------------------------------


def plot_unwrapped_phase(ds_fit: xr.Dataset, qubits) -> plt.Figure:
    """Plot unwrapped phase vs time for all qubits on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for q in qubits:
        ds_fit["phase"].sel(qubit=q.name).plot(ax=ax, label=q.name, marker=".")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.set_title("Unwrapped phase vs time")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_spectroscopy_curve(ds_fit: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Plot the measured freq-vs-flux curve(s) embedded in *ds_fit*, if present."""
    if "spec_curve_flux" not in ds_fit or "spec_curve_freq" not in ds_fit:
        return None

    source_label = ds_fit.attrs.get("freq_to_flux_sources", ds_fit.attrs.get("freq_to_flux_source", "measured"))
    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    spec_qubits = ds_fit["spec_curve_flux"].spec_qubit.values.tolist()
    for ax, qname in zip(axes[0], names):
        if qname not in spec_qubits:
            ax.set_title(f"{qname} — no curve")
            continue
        flux_arr = ds_fit["spec_curve_flux"].sel(spec_qubit=qname).values
        freq_arr = ds_fit["spec_curve_freq"].sel(spec_qubit=qname).values / 1e9
        ax.plot(flux_arr, freq_arr, lw=1.5)
        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
        ax.set_title(qname)
        ax.grid(True)
    fig.suptitle(f"Freq-vs-flux curve used ({source_label})")
    fig.tight_layout()
    return fig
