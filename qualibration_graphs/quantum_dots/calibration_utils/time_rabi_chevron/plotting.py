"""Plot chevron heatmap and FFT diagnostics (peak fit + t_π vs detuning)."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import (
    apply_qubit_outcome_style,
    empty_figure,
    qubit_success,
)
from calibration_utils.measurement_utils.measurement_streams import get_parity_item_names
from calibration_utils.time_rabi_chevron.init_utils import (
    FFT_FREQ_MIN,
    FFT_FREQ_MAX,
    compute_fft_diagnostics,
)


def _get_freq_axis_hz(ds: xr.Dataset, qubit: Any) -> np.ndarray:
    """Return drive frequency in Hz for plotting (detuning or absolute)."""
    detuning = np.asarray(ds.detuning.values, dtype=float)
    if np.abs(detuning).max() > 0.5e9:
        return detuning
    nominal = getattr(qubit.xy, "intermediate_frequency", 0.0)
    return nominal + detuning


def _get_qubit_names_from_ds(
    ds: xr.Dataset,
    qubits: List[Any],
    analysis_signal: str,
) -> List[str]:
    return get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )


def _resolve_qubit(
    qname: str,
    qubits: List[Any],
    qubits_by_name: dict,
    qubit_by_index: dict,
) -> Any | None:
    return (
        qubits_by_name.get(qname)
        or qubit_by_index.get(qname)
        or (qubits[0] if qubits else None)
    )


def _plot_chevron_ax(
    ax: plt.Axes,
    pdiff: np.ndarray,
    freq_hz: np.ndarray,
    duration_ns: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
    show_fit: bool = True,
    analysis_signal: str = "E_p1_given_p0_0",
    success: bool | None = None,
) -> None:
    """Plot a single chevron heatmap on the given axes."""
    detuning_mhz = (freq_hz - freq_hz.mean()) * 1e-6

    extent = (
        float(duration_ns[0]),
        float(duration_ns[-1]),
        float(detuning_mhz[0]),
        float(detuning_mhz[-1]),
    )
    im = ax.imshow(
        pdiff,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Drive detuning (MHz)")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Chevron data")
    plt.colorbar(im, ax=ax, label=analysis_signal)

    if fit_result and show_fit and fit_result.get("success"):
        f_res = fit_result.get("optimal_frequency", 0)
        t_pi = fit_result.get("optimal_duration", 0)
        det_res_mhz = (f_res - freq_hz.mean()) * 1e-6
        ax.axhline(det_res_mhz, color="lime", ls="--", lw=1, alpha=0.8)
        ax.axvline(t_pi, color="lime", ls="--", lw=1, alpha=0.8)
        ax.plot(t_pi, det_res_mhz, "g*", markersize=12, markeredgecolor="white")


def _plot_fft_2d_ax(
    ax: plt.Axes,
    pdiff: np.ndarray,
    freq_hz: np.ndarray,
    durations_ns: np.ndarray,
    qubit_name: str,
    f_res: float | None = None,
    success: bool | None = None,
) -> dict:
    """Plot 2D FFT heatmap with ridge fit overlaid; return diagnostics dict."""
    diag = compute_fft_diagnostics(pdiff, freq_hz, durations_ns)
    freqs_fft = diag["fft_freqs"]
    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask] * 1e3

    n_freqs = pdiff.shape[0]
    mag_2d = np.array([diag["magnitude_per_slice"][i][mask] for i in range(n_freqs)])

    detuning_mhz = (freq_hz - freq_hz.mean()) * 1e-6
    extent = (
        float(f_plot[0]),
        float(f_plot[-1]),
        float(detuning_mhz[0]),
        float(detuning_mhz[-1]),
    )
    im = ax.imshow(
        mag_2d,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="|FFT|")
    ax.set_xlabel("Rabi frequency (1/μs)")
    ax.set_ylabel("Drive detuning (MHz)")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="FFT per detuning")

    ridge = diag.get("ridge_curve_cyc_ns")
    if ridge is not None and np.any(np.isfinite(ridge)):
        ax.plot(
            ridge * 1e3,
            detuning_mhz,
            "c-",
            lw=2,
            alpha=0.9,
            label="√(Ω²+δ²)",
        )
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

    if f_res is not None:
        det_res = (f_res - freq_hz.mean()) * 1e-6
        ax.axhline(det_res, color="lime", ls="--", lw=1, alpha=0.8)

    ax.set_xlim(float(f_plot[0]), float(f_plot[-1]))
    return diag


def _plot_fft_diagnostics_panels(
    fig: Figure,
    outer_ax: plt.Axes,
    diag: dict,
    freq_hz: np.ndarray,
    qubit_name: str,
    f_res: float | None = None,
    success: bool | None = None,
) -> None:
    """Plot FFT at resonance (top) and t_π vs detuning (bottom) in one column."""
    outer_ax.axis("off")
    gs = outer_ax.get_subplotspec().subgridspec(2, 1, hspace=0.55)
    ax_fft = fig.add_subplot(gs[0])
    ax_tpi = fig.add_subplot(gs[1])

    idx = diag["resonance_idx"]
    lcurve = diag["peak_curve_per_slice"][idx]
    if lcurve is None:
        valid_idxs = [
            i
            for i in range(len(diag["peak_curve_per_slice"]))
            if diag["peak_curve_per_slice"][i] is not None
        ]
        if valid_idxs:
            idx = int(valid_idxs[np.argmin(np.abs(np.array(valid_idxs) - idx))])
            lcurve = diag["peak_curve_per_slice"][idx]
    freqs_fft = diag["fft_freqs"]
    mag = diag["magnitude_per_slice"][idx]

    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask] * 1e3
    ax_fft.plot(f_plot, mag[mask], "b-", lw=1, label="FFT")
    if lcurve is not None:
        ax_fft.plot(f_plot, lcurve[mask], "r-", lw=1.5, label="Peak fit")
    ax_fft.set_xlabel("Frequency (1/μs)")
    ax_fft.set_ylabel("|FFT|")
    apply_qubit_outcome_style(ax_fft, qubit_name, success, subtitle="FFT at resonance")
    ax_fft.legend(loc="upper right", fontsize=8)
    ax_fft.set_xlim(f_plot[0], f_plot[-1])

    t_pi = diag["t_pi_per_freq"]
    valid = np.isfinite(t_pi) & (t_pi >= 10) & (t_pi <= 500)
    detuning_mhz = (freq_hz - freq_hz.mean()) * 1e-6
    ax_tpi.scatter(
        detuning_mhz[valid],
        t_pi[valid],
        c="b",
        s=8,
        alpha=0.7,
        label="FFT peak → t_π",
    )
    rabi_curve = diag.get("rabi_curve")
    if rabi_curve is not None and np.any(np.isfinite(rabi_curve)):
        ax_tpi.plot(
            detuning_mhz,
            rabi_curve,
            "r-",
            lw=1.5,
            label="Rabi fit π/√(Ω²+δ²)",
        )
    if f_res is not None:
        det_res = (f_res - freq_hz.mean()) * 1e-6
        ax_tpi.axvline(det_res, color="lime", ls="--", lw=1, alpha=0.8)
    ax_tpi.set_xlabel("Drive detuning (MHz)")
    ax_tpi.set_ylabel("t_π (ns)")
    apply_qubit_outcome_style(ax_tpi, qubit_name, success, subtitle="t_π vs detuning")
    ax_tpi.legend(loc="upper right", fontsize=8)


def _iter_qubit_plot_context(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str,
):
    qubit_names = _get_qubit_names_from_ds(ds_fit, qubits, analysis_signal)
    qubits_by_name = {getattr(q, "name", f"Q{i}"): q for i, q in enumerate(qubits)}
    qubit_by_index = dict(
        zip(qubit_names, (qubits[i] for i in range(min(len(qubits), len(qubit_names)))))
    )
    durations_ns = np.asarray(ds_fit.pulse_duration.values, dtype=float)

    for qname in qubit_names:
        qubit = _resolve_qubit(qname, qubits, qubits_by_name, qubit_by_index)
        freq_hz = (
            _get_freq_axis_hz(ds_fit, qubit)
            if qubit
            else np.asarray(ds_fit.detuning.values, dtype=float)
        )
        fr = fit_results.get(qname, {})
        f_res = fr.get("optimal_frequency") if fr.get("success") else None
        success = qubit_success(fit_results, qname)
        signal_var = f"{analysis_signal}_{qname}"
        signal_2d = (
            np.asarray(ds_fit[signal_var].values)
            if signal_var in ds_fit.data_vars
            else None
        )
        yield qname, signal_2d, freq_hz, durations_ns, fr, f_res, success


def plot_chevron_data(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot chevron data heatmaps (one panel per qubit)."""
    contexts = list(_iter_qubit_plot_context(ds_fit, qubits, fit_results, analysis_signal))
    if not contexts:
        return empty_figure("No qubit data found in ds_fit.", figsize=(8, 4))

    n = len(contexts)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 5), squeeze=False)
    axes = axes.flatten()

    for ax, (qname, signal_2d, freq_hz, durations_ns, fr, _, success) in zip(axes, contexts):
        if signal_2d is None:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Chevron data")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue
        _plot_chevron_ax(
            ax,
            signal_2d,
            freq_hz,
            durations_ns,
            qname,
            fit_result=fr,
            show_fit=True,
            analysis_signal=analysis_signal,
            success=success,
        )

    fig.suptitle(f"Time Rabi chevron data ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_fft_2d(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot 2-D FFT heatmaps (one panel per qubit)."""
    contexts = list(_iter_qubit_plot_context(ds_fit, qubits, fit_results, analysis_signal))
    if not contexts:
        return empty_figure("No qubit data found in ds_fit.", figsize=(8, 4))

    n = len(contexts)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 5), squeeze=False)
    axes = axes.flatten()

    for ax, (qname, signal_2d, freq_hz, durations_ns, _, f_res, success) in zip(axes, contexts):
        if signal_2d is None:
            apply_qubit_outcome_style(ax, qname, success, subtitle="FFT per detuning")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue
        _plot_fft_2d_ax(ax, signal_2d, freq_hz, durations_ns, qname, f_res, success)

    fig.suptitle(f"Time Rabi chevron 2D FFT ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_fft_diagnostics(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot FFT-at-resonance and t_π-vs-detuning diagnostics (one column per qubit)."""
    contexts = list(_iter_qubit_plot_context(ds_fit, qubits, fit_results, analysis_signal))
    if not contexts:
        return empty_figure("No qubit data found in ds_fit.", figsize=(8, 4))

    n = len(contexts)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 6), squeeze=False)
    axes = axes.flatten()

    for ax, (qname, signal_2d, freq_hz, durations_ns, _, f_res, success) in zip(axes, contexts):
        if signal_2d is None:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Diagnostics")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue
        diag = compute_fft_diagnostics(signal_2d, freq_hz, durations_ns)
        _plot_fft_diagnostics_panels(fig, ax, diag, freq_hz, qname, f_res, success)

    fig.suptitle(f"Time Rabi chevron FFT diagnostics ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Dict[str, Figure]:
    """Return all standard figures for time-Rabi chevron analysis."""
    fit_results = fit_results or {}
    return {
        "chevron": plot_chevron_data(ds_fit, qubits, fit_results, analysis_signal),
        "fft_2d": plot_fft_2d(ds_fit, qubits, fit_results, analysis_signal),
        "diagnostics": plot_fft_diagnostics(ds_fit, qubits, fit_results, analysis_signal),
    }