from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style
from .analysis import _exponential_decay_model


def plot_all(
    ds_fit: xr.Dataset,
    elements: List,
    sensors: List,
    fit_results: Optional[Dict] = None,
) -> Dict[str, Figure]:
    """Plot IQ amplitude vs time with the fitted exponential decay."""
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

            if amp_key not in ds_fit:
                col += 1
                continue

            time_ns = ds_fit.time.values
            time_us = time_ns / 1e3
            amplitude = ds_fit[amp_key].values
            ax.plot(time_us, amplitude * 1e3, "o", markersize=3, label="Raw data", alpha=0.7)

            amp_corr_key = f"amplitude_corrected_{el.name}_{i + 1}"
            if amp_corr_key in ds_fit:
                amp_corr = ds_fit[amp_corr_key].values
                ax.plot(
                    time_us,
                    amp_corr * 1e3,
                    "s",
                    markersize=3,
                    color="green",
                    alpha=0.7,
                    label="With extracted correction",
                )

            success = True
            if fit_results is not None and key in fit_results:
                fit_result = fit_results[key]
                if isinstance(fit_result, dict):
                    amplitude_fit = fit_result["amplitude"]
                    tau_ns = fit_result["time_constant_ns"]
                    offset = fit_result["offset"]
                    success = bool(fit_result.get("success", True))
                else:
                    amplitude_fit = fit_result.amplitude
                    tau_ns = fit_result.time_constant_ns
                    offset = fit_result.offset
                    success = bool(getattr(fit_result, "success", True))

                if success:
                    t_fine = np.linspace(time_ns.min(), time_ns.max(), 300)
                    fit_fine = _exponential_decay_model(t_fine, amplitude_fit, tau_ns, offset)
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
            apply_sensor_outcome_style(ax, key, success)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Bias Tee Single-Shot Decay")
    fig.tight_layout()
    return {"signal_vs_time": fig}
