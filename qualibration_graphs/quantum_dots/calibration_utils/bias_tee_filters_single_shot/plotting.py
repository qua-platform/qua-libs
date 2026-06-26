from typing import List, Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .analysis import _exponential_decay_model


def plot_signal_vs_time(
    ds: xr.Dataset,
    elements: List,
    sensors: List,
    fit_results: Optional[Dict] = None,
) -> Figure:
    """Plot IQ amplitude vs time with the fitted exponential decay.

    One subplot per element/sensor combination.  Data is shown as scatter
    points and the fit (when available) as a smooth curve with the extracted
    time constant annotated.

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

            time_ns = ds.time_array.values
            time_us = time_ns / 1e3
            amplitude = ds[amp_key].values

            ax.plot(
                time_us, amplitude * 1e3, "o", markersize=3, label="Raw data", alpha=0.7
            )

            amp_corr_key = f"amplitude_corrected_{el.name}_{i + 1}"
            if amp_corr_key in ds:
                amp_corr = ds[amp_corr_key].values
                ax.plot(
                    time_us,
                    amp_corr * 1e3,
                    "s",
                    markersize=3,
                    color="green",
                    alpha=0.7,
                    label="With extracted correction",
                )

            if fit_results is not None and key in fit_results:
                fr = fit_results[key]
                if isinstance(fr, dict):
                    A = fr["amplitude"]
                    tau_ns = fr["time_constant_ns"]
                    B = fr["offset"]
                else:
                    A, tau_ns, B = fr.amplitude, fr.time_constant_ns, fr.offset

                t_fine = np.linspace(time_ns.min(), time_ns.max(), 300)
                fit_fine = _exponential_decay_model(t_fine, A, tau_ns, B)
                tau_us = tau_ns / 1e3
                ax.plot(
                    t_fine / 1e3,
                    fit_fine * 1e3,
                    "-",
                    color="red",
                    label=f"Fit: τ = {tau_us:.2f} µs ({tau_ns:.0f} ns)",
                )
                if time_ns.min() <= tau_ns <= time_ns.max():
                    ax.axvline(tau_us, color="gray", ls=":", lw=1, alpha=0.6, label="τ")

            ax.set_xlim(time_us.min(), time_us.max())
            ax.set_xlabel("Time [µs]")
            ax.set_ylabel("Amplitude [mV]")
            ax.set_title(f"{key}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Bias Tee Single-Shot Decay")
    fig.tight_layout()
    return fig
