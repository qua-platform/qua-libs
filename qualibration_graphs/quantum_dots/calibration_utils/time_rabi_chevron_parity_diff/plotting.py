"""Plot chevron heatmaps, fit surface, FFT diagnostics, and t_π vs detuning."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.time_rabi_chevron_parity_diff.init_utils import (
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


def _plot_chevron_ax(
    ax: "plt.Axes",
    pdiff: np.ndarray,
    freq_hz: np.ndarray,
    duration_ns: np.ndarray,
    qubit_name: str,
    subtitle: str = "",
    fit_result: dict | None = None,
    show_fit: bool = True,
) -> None:
    """Plot a single chevron heatmap on the given axes."""
    detuning_mhz = (freq_hz - freq_hz.mean()) * 1e-6  # Center for readability

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
    title = f"{qubit_name}" + (f" — {subtitle}" if subtitle else "")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Parity diff")

    # optimal_duration is π-time (t_π = π/Ω). Draw lines and star at π-rotation point.
    if fit_result and show_fit and fit_result.get("success"):
        f_res = fit_result.get("optimal_frequency", 0)
        t_pi = fit_result.get("optimal_duration", 0)  # π rotation duration (ns)
        det_res_mhz = (f_res - freq_hz.mean()) * 1e-6
        ax.axhline(det_res_mhz, color="lime", ls="--", lw=1, alpha=0.8)
        ax.axvline(t_pi, color="lime", ls="--", lw=1, alpha=0.8)
        ax.plot(t_pi, det_res_mhz, "g*", markersize=12, markeredgecolor="white")


def _plot_fft_diagnostics_ax(
    ax_fft: "plt.Axes",
    ax_tpi: "plt.Axes",
    pdiff: np.ndarray,
    freq_hz: np.ndarray,
    durations_ns: np.ndarray,
    qubit_name: str,
    f_res: float | None = None,
) -> None:
    """Plot FFT + peak fit (resonance slice) and t_π vs detuning + Rabi fit."""
    diag = compute_fft_diagnostics(pdiff, freq_hz, durations_ns)
    idx = diag["resonance_idx"]
    # Use resonance slice; if it has no Lorentzian fit, pick nearest slice that does
    lcurve = diag["lorentzian_curve_per_slice"][idx]
    if lcurve is None:
        valid_idxs = [
            i
            for i in range(len(diag["lorentzian_curve_per_slice"]))
            if diag["lorentzian_curve_per_slice"][i] is not None
        ]
        if valid_idxs:
            idx = int(valid_idxs[np.argmin(np.abs(np.array(valid_idxs) - idx))])
            lcurve = diag["lorentzian_curve_per_slice"][idx]
    freqs_fft = diag["fft_freqs"]
    mag = diag["magnitude_per_slice"][idx]

    # FFT + Lorentzian fit
    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask] * 1e3  # 1/ns → 1/μs for readability
    ax_fft.plot(f_plot, mag[mask], "b-", lw=1, label="FFT")
    if lcurve is not None:
        ax_fft.plot(f_plot, lcurve[mask], "r-", lw=1.5, label="Peak fit")
    ax_fft.set_xlabel("Frequency (1/μs)")
    ax_fft.set_ylabel("|FFT|")
    ax_fft.set_title(f"{qubit_name} — FFT at resonance")
    ax_fft.legend(loc="upper right", fontsize=8)
    ax_fft.set_xlim(f_plot[0], f_plot[-1])

    # t_π vs drive detuning + Rabi fit π/√(Ω²+δ²) (same validity as init: 10–500 ns)
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
    ax_tpi.set_title(f"{qubit_name} — t_π vs detuning")
    ax_tpi.legend(loc="upper right", fontsize=8)


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    """Resolve qubit names from dataset pdiff_ vars (pdiff_Q1 -> Q1)."""
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_") and not v.endswith("_fit")]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot raw data | fit | FFT+t_π for each qubit. Green marker = π-rotation point."""
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    qubits_by_name = {getattr(q, "name", f"Q{i}"): q for i, q in enumerate(qubits)}
    qubit_by_index = dict(zip(qubit_names, (qubits[i] for i in range(min(len(qubits), len(qubit_names))))))
    n = len(qubit_names)

    nrow = n
    ncol = 3  # data | fit | FFT + t_π(Rabi)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax_data, ax_fit, ax_fft_col = axes[i, 0], axes[i, 1], axes[i, 2]
        pdiff_var = f"pdiff_{qname}"
        fit_var = f"pdiff_{qname}_fit"

        qubit = qubits_by_name.get(qname) or qubit_by_index.get(qname) or (qubits[0] if qubits else None)
        freq_hz = _get_freq_axis_hz(ds, qubit) if qubit else np.asarray(ds.detuning.values, dtype=float)
        durations_ns = np.asarray(ds.pulse_duration.values, dtype=float)
        fr = fit_results.get(qname, {})

        # Left: raw data
        if pdiff_var not in ds.data_vars:
            ax_data.text(0.5, 0.5, f"No data for {qname}", transform=ax_data.transAxes, ha="center")
            ax_data.set_title(f"{qname} — data")
        else:
            pdiff = np.asarray(ds[pdiff_var].values)
            _plot_chevron_ax(ax_data, pdiff, freq_hz, durations_ns, qname, "data", fit_result=fr, show_fit=True)

        # Right: fitted output
        if ds_fit is not None and fit_var in ds_fit.data_vars:
            pdiff_fit = np.asarray(ds_fit[fit_var].values)
            _plot_chevron_ax(ax_fit, pdiff_fit, freq_hz, durations_ns, qname, "fit", fit_result=fr, show_fit=True)
        else:
            ax_fit.text(0.5, 0.5, f"No fit for {qname}", transform=ax_fit.transAxes, ha="center")
            ax_fit.set_title(f"{qname} — fit")

        # Third column: FFT + peak fit and t_π vs detuning + Rabi fit
        if pdiff_var in ds.data_vars:
            pdiff = np.asarray(ds[pdiff_var].values)
            ax_fft_col.axis("off")
            gs = ax_fft_col.get_subplotspec().subgridspec(2, 1, hspace=0.35)
            ax_fft = fig.add_subplot(gs[0])
            ax_tpi = fig.add_subplot(gs[1])
            f_res = fr.get("optimal_frequency") if fr.get("success") else None
            _plot_fft_diagnostics_ax(ax_fft, ax_tpi, pdiff, freq_hz, durations_ns, qname, f_res)
        else:
            ax_fft_col.text(0.5, 0.5, f"No data for {qname}", transform=ax_fft_col.transAxes, ha="center")

    fig.suptitle("Time Rabi chevron (parity diff)")
    fig.tight_layout()
    return fig
