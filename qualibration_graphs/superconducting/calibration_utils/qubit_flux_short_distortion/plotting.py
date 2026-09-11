"""Plotting utilities for cryoscope (17c) visualizations."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter


def _debug_suptitle(title: str) -> str:
    """Prefix debug figure titles consistently."""
    if title.startswith("Debug:"):
        return title
    return f"Debug: {title}"


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


def _plot_cryoscope_freq_ax(
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


def _plot_flux_response_ax(
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


def _plot_unwrapped_phase_ax(ax: Axes, ds: xr.Dataset, qubit: dict) -> None:
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
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default: flux response vs time (IIR fit; FIR overlay when ``fir_results`` given).

    ``log_scale=True`` uses a log time axis on the flux-response grid; otherwise linear.

    With ``debug=True``: cryoscope frequency, unwrapped phase, freq-vs-flux curve,
    and FIR resampling check.
    """
    grid_locations = [q.grid_location for q in qubits]
    figures: Dict[str, plt.Figure] = {}
    fir_by_qubit = fir_results or {}

    grid_flux = QubitGrid(ds_fit, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        qname = qubit["qubit"]
        _plot_flux_response_ax(
            ax,
            ds_fit,
            qubit,
            fit=fit_results.get(qname),
            fir_result=fir_by_qubit.get(qname),
            log_scale=log_scale,
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
        _plot_cryoscope_freq_ax(ax, ds_fit, qubit, fit=fit_results.get(qubit["qubit"]), log_scale=log_scale)
    grid_freq.fig.suptitle(_debug_suptitle("Cryoscope frequency vs time"), fontsize=16)
    grid_freq.fig.set_size_inches(15, 9)
    grid_freq.fig.tight_layout()
    figures["cryoscope_freq"] = grid_freq.fig

    if "phase" in ds_fit:
        grid_phase = QubitGrid(ds_fit, grid_locations)
        for ax, qubit in grid_iter(grid_phase):
            _plot_unwrapped_phase_ax(ax, ds_fit, qubit)
        grid_phase.fig.suptitle(_debug_suptitle("Unwrapped phase vs time"), fontsize=16)
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
            grid_curve.fig.suptitle(_debug_suptitle(f"Freq-vs-flux curve used ({source_label})"), fontsize=16)
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
            grid_fir_r.fig.suptitle(_debug_suptitle("FIR resampled flux"), fontsize=16)
            grid_fir_r.fig.set_size_inches(15, 9)
            grid_fir_r.fig.tight_layout()
            figures["fir_resampled"] = grid_fir_r.fig
        else:
            plt.close(grid_fir_r.fig)

    return figures


# ---------------------------------------------------------------------------
# Extended plotting (ported from CS_installations' node 20 plotting.py)
# ---------------------------------------------------------------------------
#
# Trimmed to the figures that add information beyond ``plot_raw_data_with_fit``'s
# flux-response grid: a raw-data view and all FIR diagnostics when debug is on.


def _qubit_names(qubits) -> list:
    """Return a list of qubit name strings regardless of input type."""
    if hasattr(qubits, "get_names"):
        return qubits.get_names()
    return [q.name if hasattr(q, "name") else str(q) for q in qubits]


def _iter_keys_and_labels(ds: xr.Dataset, qubits) -> list:
    """Return ``(sel_key, display_label)`` pairs for iterating a cryoscope dataset.

    The iteration key is always the dataset ``qubit`` coordinate value, which is
    guaranteed unique so ``.sel(qubit=key)`` collapses the ``qubit`` dimension.
    When the dataset carries a ``measured_qubit_name`` side coordinate (coupler
    nodes elsewhere in the codebase), the label becomes
    ``"<measured_qubit> (<pair>)"`` for readability; this node has no such
    coordinate so the branch is defensive/unused here.
    """
    if "qubit" in ds.coords or "qubit" in ds.dims:
        keys = [str(v) for v in ds["qubit"].values]
    else:
        keys = _qubit_names(qubits)

    has_measured = "measured_qubit_name" in ds.coords
    pairs = []
    for key in keys:
        if has_measured:
            measured = str(ds["measured_qubit_name"].sel(qubit=key).values)
            label = f"{measured} ({key})"
        else:
            label = key
        pairs.append((key, label))
    return pairs


def plot_raw_data(ds_raw: xr.Dataset, qubits) -> Dict[str, plt.Figure]:
    """Plot raw measurement data per qubit: frame slices and 2D heatmap.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with a ``state`` or ``I`` variable and dimensions
        ``(qubit, time, frame)``.
    qubits : list
        Qubit objects to plot.

    Returns
    -------
    dict
        ``{"raw_<qname>": fig, ...}`` — one figure per qubit.
    """
    figures: Dict[str, plt.Figure] = {}
    data_key = "state" if "state" in ds_raw.data_vars else "I"
    time_vals = ds_raw.time.values

    for key, label in _iter_keys_and_labels(ds_raw, qubits):
        q_data = ds_raw[data_key].sel(qubit=key)
        sample_idx = [0, len(time_vals) // 4, len(time_vals) // 2, len(time_vals) - 1]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for idx in sample_idx:
            t_sel = time_vals[idx]
            q_data.sel(time=t_sel).plot(ax=axes[0], label=f"t={t_sel} ns")
        axes[0].set_title(f"{label}: {data_key} vs frame")
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel("Frame")
        axes[0].grid(True)
        frame_vals = q_data.frame.values if "frame" in ds_raw.dims else np.linspace(0, 1, q_data.shape[1])
        im = axes[1].pcolormesh(
            q_data.time.values,
            frame_vals,
            q_data.values.T,
            shading="auto",
            cmap="viridis",
        )
        fig.colorbar(im, ax=axes[1]).set_label(data_key.capitalize())
        axes[1].set_title(f"{label}: {data_key}(time, frame)")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel("Frame")
        fig.suptitle(_debug_suptitle(f"Raw {data_key} — {label}"), y=1.02)
        fig.tight_layout()
        figures[f"raw_{key}"] = fig

    return figures


def _plot_fir_forward_diagnostic(res: dict, qname: str) -> plt.Figure:
    """2×2 forward-FIR diagnostic (auto-tuned path)."""
    h = np.asarray(res["forward_fir"], dtype=float)
    response = np.asarray(res["normalized_2gs"], dtype=float)
    time = np.asarray(res["time_2gs"], dtype=float)
    reconstructed = np.asarray(res["reconstructed_2gs"], dtype=float)
    chosen_l = res.get("auto_chosen_L", res.get("L"))
    forward_search = res.get("forward_search", {})

    residual = response - reconstructed
    nrms = float(np.linalg.norm(residual) / max(np.linalg.norm(response), 1e-30))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(time, response, "r-", lw=2, alpha=0.7, label="Measured (distorted)")
    ax.plot(time, reconstructed, "b--", lw=2, label=f"Reconstructed (NRMS={nrms:.4e})")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Best Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time, residual, "m-", lw=1.5)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.fill_between(time, -np.std(residual), np.std(residual), alpha=0.2, color="gray")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Reconstruction Residual (σ={np.std(residual):.4e})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(h, "b-o", ms=4, lw=2)
    ax.set_xlabel("Tap Index")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Forward FIR (L={chosen_l}, auto-chosen by AIC)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    colors = {"gcv": "steelblue", "lcurve": "darkorange"}
    for crit, records in forward_search.items():
        ls = [r["L"] for r in records]
        nrmss = [r["nrms"] for r in records]
        ax.plot(ls, nrmss, "o-", lw=2, ms=6, label=crit, color=colors.get(crit))
    ax.axvline(chosen_l, color="r", ls="--", lw=1.5, alpha=0.7, label=f"Chosen L={chosen_l} (AIC)")
    ax.axhline(nrms, color="r", ls=":", alpha=0.5, label=f"Best NRMS: {nrms:.4e}")
    ax.set_xlabel("Filter Length L")
    ax.set_ylabel("NRMS (optimal λ per L)")
    ax.set_title("NRMS vs Filter Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(_debug_suptitle(f"FIR forward fit diagnostic — {qname}"), fontsize=14)
    fig.tight_layout()
    return fig


def _plot_fir_inverse_diagnostic(res: dict, qname: str, *, Ts: float = 0.5) -> plt.Figure:
    """3×2 inverse-FIR and correction diagnostic."""
    time = np.asarray(res["time_2gs"], dtype=float)
    response = np.asarray(res["normalized_2gs"], dtype=float)
    best_reconstructed = np.asarray(res["reconstructed_2gs"], dtype=float)
    h_fwd = np.asarray(res["forward_fir"], dtype=float)
    h_inv = np.asarray(res["inverse_fir"], dtype=float)
    ideal_response = np.asarray(res["ideal_response"], dtype=float)
    predistorted_response = np.asarray(res["predistorted_response"], dtype=float)
    corrected_response = np.asarray(res["corrected_response"], dtype=float)
    corrected_from_measured = np.asarray(res["corrected_from_measured"], dtype=float)
    delta = np.asarray(res["delta"], dtype=float)
    res_fit = np.asarray(res["res_fit"], dtype=float)
    res_corr = np.asarray(res["res_corr"], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, response, "r-", lw=2, label="Distorted")
    ax.plot(time, best_reconstructed, "b:", lw=2, alpha=0.7, label="FIR model")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signals and FIR Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(h_fwd, "b-o", ms=4, lw=2, label="Forward h")
    ax.plot(h_inv, "r-s", ms=4, lw=2, label="Inverse h_inv")
    ax.set_xlabel("Tap Index")
    ax.set_ylabel("Coefficient")
    ax.set_title("FIR Filters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, predistorted_response, "c-", lw=2, alpha=0.7, label="Predistorted")
    ax.plot(time, corrected_response, "m-", lw=2, label="Corrected (sim)")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Predistortion and Correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time, ideal_response, "g--", lw=2, alpha=0.7, label="Ideal")
    ax.plot(time, response, "r-", lw=2, alpha=0.5, label="Distorted")
    ax.plot(time, corrected_response, "m-", lw=2, label="Corrected (predistort)")
    ax.plot(time, corrected_from_measured, color="orange", lw=2, label="Corrected (measured)")
    ax.axhline(1.001, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.999, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim([0.95, 1.05])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Correction Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(time, res_fit, "b-", lw=1.5, label=f"Fit residual (σ={np.std(res_fit):.4e})")
    ax.plot(time, res_corr, "m-", lw=1.5, label=f"Correction residual (σ={np.std(res_corr):.4e})")
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(np.arange(len(delta)) * Ts, delta, "g-o", ms=4, lw=2)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"h * h_inv ≈ δ  (peak={np.max(np.abs(delta)):.3e})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(_debug_suptitle(f"FIR inverse / correction diagnostic — {qname}"), fontsize=14)
    fig.tight_layout()
    return fig


def plot_fir_figures(
    ds_fit: xr.Dataset,
    qubits,
    fir_results: dict,
    *,
    debug: bool = False,
) -> Dict[str, plt.Figure]:
    """Plot FIR diagnostic figures for each qubit (``debug=True`` only).

    When ``debug=True`` and FIR analysis succeeded, generates per qubit:
      - ``fir_fit_diagnostic_<qname>``: forward FIR 2×2 (reconstruction, NRMS vs L).
      - ``fir_corrected_<qname>``: corrected step response at 1 GS/s.
      - ``fir_inverse_diagnostic_<qname>``: inverse FIR 3×2 correction diagnostic.
      - ``fir_stem_<qname>``: forward / inverse tap stem plots.

    Returns an empty dict when ``debug=False``.
    """
    if not debug:
        return {}

    figures: Dict[str, plt.Figure] = {}
    for key, label in _iter_keys_and_labels(ds_fit, qubits):
        qname = key
        res = fir_results.get(qname)
        if res is None or not res.get("success"):
            continue

        t1 = np.array(res["time_1gs"])

        figures[f"fir_fit_diagnostic_{qname}"] = _plot_fir_forward_diagnostic(res, label)
        figures[f"fir_inverse_diagnostic_{qname}"] = _plot_fir_inverse_diagnostic(res, label)

        fig7, ax7 = plt.subplots(figsize=(10, 5))
        ax7.plot(t1, res["normalized_1gs"], label="data (normalized)")
        ax7.plot(t1, res["corrected_1gs"], "--", label="expected corrected response")
        ax7.axhline(1.001, color="k", lw=0.8, ls="--", label="±0.1% tolerance")
        ax7.axhline(0.999, color="k", lw=0.8, ls="--")
        ax7.set_ylim([0.95, 1.05])
        sigma_disp = res.get("noise_sigma_displayed")
        noise_msg = res.get("noise_estimate_msg")
        if sigma_disp is not None and noise_msg is not None:
            ax7.plot([], [], " ", label=f"noise σ≈{sigma_disp:.1e} [{noise_msg}]")
        ax7.legend()
        ax7.set_xlabel("Time (ns)")
        ax7.set_ylabel("Normalized amplitude")
        ax7.grid(True, alpha=0.3)
        fig7.suptitle(_debug_suptitle(f"FIR corrected step response — {label}"), fontsize=14)
        fig7.tight_layout()
        figures[f"fir_corrected_{qname}"] = fig7

        h_fir_arr = np.array(res["forward_fir"])
        h_inv_arr = np.array(res["inverse_fir"])
        fig8, axes8 = plt.subplots(1, 2, figsize=(14, 4))
        axes8[0].stem(np.arange(len(h_fir_arr)), h_fir_arr, linefmt="b-", markerfmt="bo", basefmt="k-")
        axes8[0].set_xlabel("Tap Index")
        axes8[0].set_ylabel("Coefficient")
        axes8[0].set_title(f"Forward FIR h (L={len(h_fir_arr)})")
        axes8[0].grid(True, alpha=0.3)
        axes8[1].stem(np.arange(len(h_inv_arr)), h_inv_arr, linefmt="r-", markerfmt="rs", basefmt="k-")
        axes8[1].set_xlabel("Tap Index")
        axes8[1].set_ylabel("Coefficient")
        axes8[1].set_title(f"Inverse FIR h_inv (M={len(h_inv_arr)})")
        axes8[1].grid(True, alpha=0.3)
        fig8.suptitle(_debug_suptitle(f"FIR filter coefficients — {label}"), fontsize=14)
        fig8.tight_layout()
        figures[f"fir_stem_{qname}"] = fig8

    return figures
