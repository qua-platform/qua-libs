from typing import List, Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .analysis import _high_pass_model


def plot_signal_vs_frequency(
    ds: xr.Dataset,
    elements: List,
    sensors: List,
    fit_results: Optional[Dict] = None,
) -> Figure:
    """Plot IQ amplitude vs square-wave frequency with the fitted high-pass curve.

    One subplot per element/sensor combination.  Data is shown as scatter
    points and the fit (when available) as a smooth curve with the extracted
    cutoff frequency and time constant annotated.

    Args:
        ds: Dataset containing ``amplitude_*`` and optionally ``fit_*`` variables.
        elements: List of element objects from ``node.namespace["elements"]``.
        sensors: List of sensor objects from ``node.namespace["sensors"]``.
        fit_results: Optional dict of serialised ``FitParameters`` keyed by
            ``{el_name}_{sensor_name}``.

    Returns:
        Matplotlib Figure.
    """
    n_cols = len(elements) * len(sensors)
    fig, axes = plt.subplots(
        1,
        max(n_cols, 1),
        figsize=(6 * max(n_cols, 1), 5),
        squeeze=False,
    )
    axes = axes.flatten()

    col = 0
    for el in elements:
        for i, sensor in enumerate(sensors):
            ax = axes[col]
            key = f"{el.name}_{sensor.name}"
            amp_key = f"amplitude_{el.name}_{i + 1}"

            if amp_key not in ds:
                col += 1
                continue

            freqs = ds.frequency.values
            amplitude = ds[amp_key].values

            freqs_kHz = freqs / 1e3
            ax.plot(
                freqs_kHz,
                amplitude * 1e3,
                "o",
                markersize=4,
                label="Raw data",
                alpha=0.7,
            )

            amp_corr_key = f"amplitude_corrected_{el.name}_{i + 1}"
            if amp_corr_key in ds:
                amp_corr = ds[amp_corr_key].values
                ax.plot(
                    freqs_kHz,
                    amp_corr * 1e3,
                    "s",
                    markersize=4,
                    color="green",
                    alpha=0.7,
                    label="With extracted correction",
                )

            if fit_results is not None and key in fit_results:
                fr = fit_results[key]
                if isinstance(fr, dict):
                    A = fr["amplitude"]
                    f_c = fr["cutoff_frequency_Hz"]
                    B = fr["offset"]
                    tau_ns = fr["time_constant_ns"]
                else:
                    A, f_c, B = fr.amplitude, fr.cutoff_frequency_Hz, fr.offset
                    tau_ns = fr.time_constant_ns

                f_fine = np.linspace(freqs.min(), freqs.max(), 300)
                fit_fine = _high_pass_model(f_fine, A, f_c, B)
                ax.plot(
                    f_fine / 1e3,
                    fit_fine * 1e3,
                    "-",
                    color="red",
                    label=f"Fit: τ = {tau_ns:.0f} ns, f_c = {f_c / 1e3:.1f} kHz",
                )
                if freqs.min() <= f_c <= freqs.max():
                    ax.axvline(f_c / 1e3, color="gray", ls=":", lw=1, alpha=0.6)

            ax.set_xlim(freqs_kHz.min(), freqs_kHz.max())
            ax.set_xlabel("Frequency [kHz]")
            ax.set_ylabel("Amplitude [mV]")
            ax.set_title(f"{key}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Bias Tee Filter Characterization")
    fig.tight_layout()
    return fig
