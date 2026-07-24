"""Plot power-Rabi conditional expectation: raw trace and FFT diagnostics."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.power_rabi.analysis import FFT_FREQ_MIN, FFT_FREQ_MAX, compute_fft_diagnostic


def _get_qubit_names_from_ds(
    ds: xr.Dataset,
    analysis_signal: str = "E_p1_given_p0_0",
) -> List[str]:
    signal_prefix = f"{analysis_signal}_"
    signal_vars = [
        v
        for v in ds.data_vars
        if v.startswith(signal_prefix) and not v.endswith("_fit")
    ]
    if signal_vars:
        return [v.replace(signal_prefix, "") for v in sorted(signal_vars)]

    p0_p0_vars = [v for v in ds.data_vars if v.startswith("p0_p0_")]
    if p0_p0_vars:
        return [v.replace("p0_p0_", "") for v in sorted(p0_p0_vars)]
    names: List[str] = []
    for v in sorted(ds.data_vars):
        if v.startswith("p_") and not v.startswith(("p0_", "p1_", "pdiff_", "E_")):
            rest = v[2:]
            if rest:
                names.append(rest)
    return names


def _reference_amplitude(qubit: Any) -> float:
    """Return the calibrated XY reference amplitude used for power-Rabi scaling."""
    try:
        xy_drive = qubit.macros.get("xy_drive")
        if xy_drive is not None and hasattr(xy_drive, "reference_amplitude"):
            return float(xy_drive.reference_amplitude)
    except Exception:
        pass
    return 1.0


def _add_prefactor_top_axis(
    ax: "plt.Axes",
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
    ax: "plt.Axes",
    trace: np.ndarray,
    prefactors: np.ndarray,
    qubit_name: str,
    analysis_signal: str,
    fit_result: dict | None = None,
    fitted_curve: np.ndarray | None = None,
    reference_amplitude: float = 1.0,
) -> None:
    """Plot analysis trace vs pulse amplitude with prefactor on the top axis."""
    pulse_amps = np.asarray(prefactors, dtype=float) * reference_amplitude
    ax.plot(pulse_amps, trace, "b-", lw=1, alpha=0.8)
    ax.scatter(pulse_amps, trace, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Pulse amplitude")
    ax.set_ylabel(analysis_signal)
    ax.set_title(f"{qubit_name} — Power Rabi")
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
    ax: "plt.Axes",
    qubit_name: str,
    trace: np.ndarray,
    amps: np.ndarray,
    fit_result: dict | None = None,
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
    ax.set_title(f"{qubit_name} — FFT spectrum")
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


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
    parity_measurement: bool | None = None,
) -> "plt.Figure":
    """Plot power-Rabi trace and FFT for each qubit.

    Layout (per qubit row):
    * Column 1 — Conditional expectation vs pulse amplitude (bottom axis)
      with amplitude prefactor on the top axis.
    * Column 2 — FFT magnitude spectrum with peak fit overlay.

    ``ds`` should be the processed dataset (``ds_fit``) containing
    ``{analysis_signal}_{qubit}`` variables.
    """
    plot_ds = ds_fit if ds_fit is not None else ds
    qubit_names = _get_qubit_names_from_ds(plot_ds, analysis_signal)
    qubits_by_name = {getattr(q, "name", str(i)): q for i, q in enumerate(qubits)}
    if not qubit_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No qubit data found in ds_fit.\n"
            "Check that generate_simulated_data / analyse_data ran successfully\n"
            "and that node.parameters.qubits (or active_qubit_names) is set.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        return fig

    if parity_measurement is None:
        parity_measurement = any(
            v.startswith(f"{analysis_signal}_") or v.startswith("p0_p0_")
            for v in plot_ds.data_vars
        )

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax_trace, ax_fft = axes[i, 0], axes[i, 1]
        if f"{analysis_signal}_{qname}" in plot_ds.data_vars:
            signal_var = f"{analysis_signal}_{qname}"
            y_signal_label = analysis_signal
        else:
            signal_var = f"p_{qname}"
            y_signal_label = "P(measure)"
        fr = fit_results.get(qname, {})

        amps = np.asarray(plot_ds.amp_prefactor.values, dtype=float)
        qubit = qubits_by_name.get(qname)
        ref_amp = _reference_amplitude(qubit) if qubit is not None else 1.0

        if signal_var not in plot_ds.data_vars:
            ax_trace.text(
                0.5,
                0.5,
                f"No data for {qname}",
                transform=ax_trace.transAxes,
                ha="center",
            )
            ax_fft.text(
                0.5,
                0.5,
                f"No data for {qname}",
                transform=ax_fft.transAxes,
                ha="center",
            )
            continue

        trace = np.asarray(plot_ds[signal_var].values, dtype=float)
        fit_var = f"{signal_var}_fit"
        fitted_curve = None
        if plot_ds is not None and fit_var in plot_ds.data_vars:
            fitted_curve = np.asarray(plot_ds[fit_var].values, dtype=float)

        _plot_rabi_trace_ax(
            ax_trace,
            trace,
            amps,
            qname,
            y_signal_label,
            fit_result=fr,
            fitted_curve=fitted_curve,
            reference_amplitude=ref_amp,
        )
        _plot_fft_ax(ax_fft, qname, trace, amps, fit_result=fr)

    fig.suptitle(
        f"Power Rabi ({analysis_signal})"
        if parity_measurement
        else "Power Rabi (single measurement)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig
