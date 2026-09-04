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
    fit_text = "IIR fit:\n"
    fit_text += f"a_dc = {a_dc:.3f}\n"
    y_fit = np.ones_like(t_data, dtype=float) * a_dc
    for i, (amp, tau) in enumerate(components):
        y_fit += amp * np.exp(-t_data / tau)
        fit_text += f"a{i + 1} = {amp / a_dc:.3f}, τ{i + 1} = {tau:.0f}ns\n"
    return y_fit, fit_text


def _flux_fit_to_freq(freq: np.ndarray, flux: np.ndarray, flux_fit: np.ndarray) -> np.ndarray:
    """Map a fitted flux trace back to frequency via the empirical quadratic scale."""
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
    """Plot cryoscope frequency vs time on *ax*, optionally with the IIR model."""
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
    fir_result: Optional[dict] = None,
    log_scale: bool = False,
) -> None:
    """Plot flux step response vs time on *ax*, with optional IIR and FIR overlays."""
    qname = qubit["qubit"]
    t_data = np.asarray(ds.time.values, dtype=float)
    flux = np.asarray(ds["flux_response"].sel(qubit=qname).values, dtype=float)
    ax.plot(t_data, flux, ".--", label="Data")

    if fit is not None:
        components, a_dc = _unpack_fit(fit, flux)
        if components and np.isfinite(a_dc):
            flux_fit, fit_text = _exp_fit_curve(t_data, components, a_dc)
            ax.plot(t_data, flux_fit, "-", label="IIR fit")
            ax.text(
                0.98,
                0.5,
                fit_text,
                transform=ax.transAxes,
                fontsize=12,
                horizontalalignment="right",
                verticalalignment="center",
            )

    if fir_result is not None and fir_result.get("success"):
        tail_mean = float(np.nanmean(flux[-10:])) or 1.0
        corrected = np.asarray(fir_result["corrected_1gs"], dtype=float) * tail_mean
        ax.plot(t_data, corrected, "--", label="FIR-corrected")

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


def plot_unwrapped_phase(ax: Axes, ds: xr.Dataset, qubit: dict) -> None:
    """Plot unwrapped Ramsey phase vs time on one axis."""
    qname = qubit["qubit"]
    if "phase" not in ds:
        ax.set_title(f"{qname} — no phase")
        return
    ds["phase"].sel(qubit=qname).plot(ax=ax, marker=".")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.set_title(qname)
    ax.grid(True)


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


def plot_fir_resampled(ax: Axes, fir_result: dict) -> None:
    """Plot 1 GS/s vs 2 GS/s resampled flux used for FIR fitting."""
    t1 = np.asarray(fir_result["time_1gs"])
    t2 = np.asarray(fir_result["time_2gs"])
    ax.plot(t1, fir_result["normalized_1gs"], "b.-", label="1 GS/s", alpha=0.6)
    ax.plot(t2, fir_result["normalized_2gs"], "r.-", ms=3, label="2 GS/s", alpha=0.6)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubits,
    fit_results: dict,
    *,
    debug: bool = False,
    fir_results: Optional[dict] = None,
) -> Dict[str, plt.Figure]:
    """Default: flux response vs time (IIR fit; FIR overlay when ``fir_results`` given).

    With ``debug=True``: cryoscope frequency, unwrapped phase, freq-vs-flux curve,
    and FIR resampling check.
    """
    grid_locations = [q.grid_location for q in qubits]
    figures: Dict[str, plt.Figure] = {}
    fir_by_qubit = fir_results or {}

    grid_flux = QubitGrid(ds_fit, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        qname = qubit["qubit"]
        plot_flux_response(
            ax,
            ds_fit,
            qubit,
            fit=fit_results.get(qname),
            fir_result=fir_by_qubit.get(qname),
        )
    title = "Flux response vs time"
    if any(res.get("success") for res in fir_by_qubit.values()):
        title += " (with FIR overlay)"
    grid_flux.fig.suptitle(title, fontsize=16)
    grid_flux.fig.set_size_inches(15, 9)
    grid_flux.fig.tight_layout()
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    grid_freq = QubitGrid(ds_fit, grid_locations)
    for ax, qubit in grid_iter(grid_freq):
        plot_cryoscope_freq(ax, ds_fit, qubit, fit=fit_results.get(qubit["qubit"]))
    grid_freq.fig.suptitle("Debug: Cryoscope frequency vs time", fontsize=16)
    grid_freq.fig.set_size_inches(15, 9)
    grid_freq.fig.tight_layout()
    figures["cryoscope_freq"] = grid_freq.fig

    if "phase" in ds_fit:
        grid_phase = QubitGrid(ds_fit, grid_locations)
        for ax, qubit in grid_iter(grid_phase):
            plot_unwrapped_phase(ax, ds_fit, qubit)
        grid_phase.fig.suptitle("Debug: Unwrapped phase vs time", fontsize=16)
        grid_phase.fig.set_size_inches(15, 9)
        grid_phase.fig.tight_layout()
        figures["unwrapped_phase"] = grid_phase.fig

    if "spec_curve_flux" in ds_fit and "spec_curve_freq" in ds_fit:
        source_label = ds_fit.attrs.get("freq_to_flux_sources", ds_fit.attrs.get("freq_to_flux_source", "measured"))
        grid_curve = QubitGrid(ds_fit, grid_locations)
        n_plotted = 0
        for ax, qubit in grid_iter(grid_curve):
            plot_freq_vs_flux_curve(ax, ds_fit, qubit, source_label=source_label)
            qname = qubit["qubit"]
            if qname in ds_fit["spec_curve_flux"].spec_qubit.values.tolist():
                flux_arr = ds_fit["spec_curve_flux"].sel(spec_qubit=qname).values
                freq_arr = ds_fit["spec_curve_freq"].sel(spec_qubit=qname).values
                if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                    n_plotted += 1
        if n_plotted:
            grid_curve.fig.suptitle(f"Debug: Freq-vs-flux curve used ({source_label})", fontsize=16)
            grid_curve.fig.set_size_inches(15, 9)
            grid_curve.fig.tight_layout()
            figures["freq_vs_flux_curve"] = grid_curve.fig
        else:
            plt.close(grid_curve.fig)

    if fir_results:
        grid_fir_r = QubitGrid(ds_fit, grid_locations)
        n_fir = 0
        for ax, qubit in grid_iter(grid_fir_r):
            qname = qubit["qubit"]
            res = fir_results.get(qname)
            if res is not None and res.get("success"):
                plot_fir_resampled(ax, res)
                ax.set_title(qname)
                n_fir += 1
            else:
                ax.set_title(f"{qname} — no FIR")
        if n_fir:
            grid_fir_r.fig.suptitle("Debug: FIR resampled flux", fontsize=16)
            grid_fir_r.fig.set_size_inches(15, 9)
            grid_fir_r.fig.tight_layout()
            figures["fir_resampled"] = grid_fir_r.fig
        else:
            plt.close(grid_fir_r.fig)

    return figures


def plot_spectroscopy_curve(ds_fit: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Legacy multi-qubit freq-vs-flux figure; prefer ``plot_raw_data_with_fit``."""
    if "spec_curve_flux" not in ds_fit or "spec_curve_freq" not in ds_fit:
        return None
    source_label = ds_fit.attrs.get("freq_to_flux_sources", ds_fit.attrs.get("freq_to_flux_source", "measured"))
    grid_locations = [q.grid_location for q in qubits]
    grid_curve = QubitGrid(ds_fit, grid_locations)
    n_plotted = 0
    for ax, qubit in grid_iter(grid_curve):
        plot_freq_vs_flux_curve(ax, ds_fit, qubit, source_label=source_label)
        qname = qubit["qubit"]
        if qname in ds_fit["spec_curve_flux"].spec_qubit.values.tolist():
            flux_arr = ds_fit["spec_curve_flux"].sel(spec_qubit=qname).values
            freq_arr = ds_fit["spec_curve_freq"].sel(spec_qubit=qname).values
            if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                n_plotted += 1
    if not n_plotted:
        plt.close(grid_curve.fig)
        return None
    grid_curve.fig.suptitle(f"Debug: Freq-vs-flux curve used ({source_label})", fontsize=16)
    grid_curve.fig.tight_layout()
    return grid_curve.fig
