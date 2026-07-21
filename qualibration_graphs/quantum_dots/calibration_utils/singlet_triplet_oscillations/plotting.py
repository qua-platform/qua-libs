from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _plot_trace_with_fit(
    ax: plt.Axes,
    wait_ns: np.ndarray,
    trace: np.ndarray,
    title: str,
    ylabel: str,
    fit_trace: np.ndarray | None = None,
) -> None:
    ax.plot(wait_ns, trace, "o-", label="data", markersize=4)
    if fit_trace is not None:
        ax.plot(wait_ns, fit_trace, "-", label="fit", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Wait duration (ns)")
    ax.set_ylabel(ylabel)
    ax.legend()


def _plot_fft(ax: plt.Axes, wait_ns: np.ndarray, trace: np.ndarray, title: str) -> None:
    centered = trace - np.mean(trace)
    dt_s = float(wait_ns[1] - wait_ns[0]) * 1e-9 if len(wait_ns) > 1 else 1e-9
    spectrum = np.abs(np.fft.rfft(centered))
    freqs_mhz = np.fft.rfftfreq(len(centered), d=dt_s) * 1e-6

    if spectrum.size > 0:
        spectrum[0] = 0.0
    ax.plot(freqs_mhz, spectrum, "-")
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|FFT|")


def plot_singlet_triplet_oscillations(
    ds_raw: xr.Dataset,
    qubit_pair_name: str,
    *,
    ds_fit: xr.Dataset | None = None,
) -> plt.Figure:
    """Plot singlet-triplet traces (state/I/Q) and their FFT spectra."""
    wait_ns = ds_raw.coords["wait_duration"].values
    state = ds_raw[f"state_{qubit_pair_name}"].values
    i_signal = ds_raw[f"I_{qubit_pair_name}"].values
    q_signal = ds_raw[f"Q_{qubit_pair_name}"].values

    fit_trace = None
    if ds_fit is not None and "model" in ds_fit:
        fit_trace = ds_fit["model"].values

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    _plot_trace_with_fit(
        axes[0, 0],
        wait_ns,
        state,
        f"{qubit_pair_name} - state oscillations",
        "Average state",
        fit_trace=fit_trace,
    )
    _plot_fft(axes[0, 1], wait_ns, state, f"{qubit_pair_name} - FFT(state)")

    _plot_trace_with_fit(
        axes[1, 0],
        wait_ns,
        i_signal,
        f"{qubit_pair_name} - I oscillations",
        "Average I",
    )
    _plot_fft(axes[1, 1], wait_ns, i_signal, f"{qubit_pair_name} - FFT(I)")

    _plot_trace_with_fit(
        axes[2, 0],
        wait_ns,
        q_signal,
        f"{qubit_pair_name} - Q oscillations",
        "Average Q",
    )
    _plot_fft(axes[2, 1], wait_ns, q_signal, f"{qubit_pair_name} - FFT(Q)")

    fig.suptitle("Singlet-triplet oscillations")
    fig.tight_layout()
    return fig
