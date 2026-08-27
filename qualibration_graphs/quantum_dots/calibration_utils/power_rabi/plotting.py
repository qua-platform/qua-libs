"""Plot power-Rabi conditional expectation: raw trace and FFT diagnostics."""

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
from calibration_utils.power_rabi.analysis import FFT_FREQ_MIN, FFT_FREQ_MAX, compute_fft_diagnostic


def _reference_amplitude(qubit: Any) -> float:
    """Return the calibrated XY reference amplitude used for power-Rabi scaling."""
    try:
        xy_drive = qubit.macros.get("xy_drive")
        if xy_drive is not None and hasattr(xy_drive, "reference_amplitude"):
            return float(xy_drive.reference_amplitude)
    except Exception:
        pass
    return 1.0


def _resolve_signal_var(plot_ds: xr.Dataset, qname: str, analysis_signal: str) -> tuple[str, str]:
    if f"{analysis_signal}_{qname}" in plot_ds.data_vars:
        return f"{analysis_signal}_{qname}", analysis_signal
    return f"p_{qname}", "P(measure)"


def _add_prefactor_top_axis(
    ax: plt.Axes,
    prefactors: np.ndarray,
    reference_amplitude: float,
) -> None:
    """Add a top x-axis labelled in amplitude prefactor units."""
    prefactors = np.asarray(prefactors, dtype=float)
    amp_ticks = np.linspace(prefactors.min(), prefactors.max(), num=5) * reference_amplitude
    pref_labels = amp_ticks / reference_amplitude
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(amp_ticks)
    ax_top.set_xticklabels([f"{p:.3g}" for p in pref_labels])
    ax_top.set_xlabel("Amplitude prefactor")


def _plot_rabi_trace_ax(
    ax: plt.Axes,
    trace: np.ndarray,
    prefactors: np.ndarray,
    qubit_name: str,
    analysis_signal: str,
    fit_result: dict | None = None,
    fitted_curve: np.ndarray | None = None,
    reference_amplitude: float = 1.0,
    success: bool | None = None,
) -> None:
    """Plot analysis trace vs pulse amplitude with prefactor on the top axis."""
    pulse_amps = np.asarray(prefactors, dtype=float) * reference_amplitude
    ax.plot(pulse_amps, trace, "b-", lw=1, alpha=0.8)
    ax.scatter(pulse_amps, trace, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Pulse amplitude")
    ax.set_ylabel(analysis_signal)
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Power Rabi")
    ax.set_ylim(-0.05, 1.05)

    if fit_result and fit_result.get("success"):
        a_pi = fit_result.get("opt_amp", 0)

        if fitted_curve is not None:
            ax.plot(
                pulse_amps,
                fitted_curve,
                "r-",
                lw=1.5,
                alpha=0.9,
                label="Damped sinusoid fit",
            )

        ax.axvline(
            a_pi * reference_amplitude,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=f"a_π = {a_pi:.3f}",
        )
        ax.legend(loc="upper right", fontsize=8)

    _add_prefactor_top_axis(ax, prefactors, reference_amplitude)


def _plot_fft_ax(
    ax: plt.Axes,
    qubit_name: str,
    trace: np.ndarray,
    amps: np.ndarray,
    fit_result: dict | None = None,
    success: bool | None = None,
) -> None:
    """Plot FFT magnitude spectrum with peak fit."""
    diag = compute_fft_diagnostic(trace, amps)
    freqs_fft = diag["fft_freqs"]
    magnitude = diag["fft_magnitude"]
    peak_curve = diag.get("peak_curve")

    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask]

    ax.plot(f_plot, magnitude[mask], "b-", lw=1, label="FFT")
    if peak_curve is not None:
        ax.plot(f_plot, peak_curve[mask], "r-", lw=1.5, label="Peak fit")

    ax.set_xlabel("Frequency (cycles / unit amp)")
    ax.set_ylabel("|FFT|")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="FFT spectrum")
    ax.set_xlim(f_plot[0], f_plot[-1])

    if fit_result and fit_result.get("success"):
        omega = fit_result.get("rabi_frequency", 0)
        f_rabi = omega / (2.0 * np.pi)
        ax.axvline(
            f_rabi,
            color="lime",
            ls="--",
            lw=1,
            alpha=0.9,
            label=f"f = {f_rabi:.2f} c/u.a.",
        )

    ax.legend(loc="upper right", fontsize=8)


def _empty_message() -> str:
    return (
        "No qubit data found in ds_fit.\n"
        "Check that generate_simulated_data / analyse_data ran successfully\n"
        "and that node.parameters.qubits (or active_qubit_names) is set."
    )


def plot_rabi_traces(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot power-Rabi traces with fit overlays (one panel per qubit)."""
    qubit_names = get_parity_item_names(
        ds_fit,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )
    if not qubit_names:
        return empty_figure(_empty_message())

    qubits_by_name = {getattr(q, "name", str(i)): q for i, q in enumerate(qubits)}
    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)
    for ax, qname in zip(axes, qubit_names):
        signal_var, y_label = _resolve_signal_var(ds_fit, qname, analysis_signal)
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Power Rabi")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        qubit = qubits_by_name.get(qname)
        ref_amp = _reference_amplitude(qubit) if qubit is not None else 1.0
        trace = np.asarray(ds_fit[signal_var].values, dtype=float)
        fit_var = f"{signal_var}_fit"
        fitted_curve = np.asarray(ds_fit[fit_var].values, dtype=float) if fit_var in ds_fit.data_vars else None
        _plot_rabi_trace_ax(
            ax,
            trace,
            amps,
            qname,
            y_label,
            fit_result=fr,
            fitted_curve=fitted_curve,
            reference_amplitude=ref_amp,
            success=success,
        )

    parity_measurement = any(v.startswith(f"{analysis_signal}_") or v.startswith("p0_p0_") for v in ds_fit.data_vars)
    fig.suptitle(f"Power Rabi ({analysis_signal})" if parity_measurement else "Power Rabi (single measurement)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_fft_spectra(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot FFT magnitude spectra (one panel per qubit)."""
    qubit_names = get_parity_item_names(
        ds_fit,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )
    if not qubit_names:
        return empty_figure(_empty_message())

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)
    for ax, qname in zip(axes, qubit_names):
        signal_var, _ = _resolve_signal_var(ds_fit, qname, analysis_signal)
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="FFT spectrum")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        trace = np.asarray(ds_fit[signal_var].values, dtype=float)
        _plot_fft_ax(ax, qname, trace, amps, fit_result=fr, success=success)

    fig.suptitle(f"Power Rabi FFT ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Dict[str, Figure]:
    """Return all standard figures for power-Rabi analysis."""
    fit_results = fit_results or {}
    return {
        "rabi": plot_rabi_traces(ds_fit, qubits, fit_results, analysis_signal),
        "fft": plot_fft_spectra(ds_fit, qubits, fit_results, analysis_signal),
    }
