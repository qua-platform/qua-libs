from typing import List, Dict
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from qualang_tools.units import unit

from .analysis import _fit_components_for_plot

u = unit(coerce_to_integer=True)


def plot_iq_histogram(
    ds: xr.Dataset,
    all_sensors: Dict,
    quantum_dot_pairs: List,
    fit_results: Dict | None = None,
) -> Figure:
    """2D IQ histograms at the longest integration time for each sensor/dot-pair."""
    n_cols = sum(len(all_sensors[dp.name]) for dp in quantum_dot_pairs)
    fig, axes = plt.subplots(
        1, max(n_cols, 1), figsize=(6 * max(n_cols, 1), 5), squeeze=False
    )
    axes = axes.flatten()

    col = 0
    for dp in quantum_dot_pairs:
        for sensor in all_sensors[dp.name]:
            key = f"{dp.name}_{sensor.name}"
            ax = axes[col]

            I_11 = ds[f"I_11_{key}"].isel(integration_time=-1).values.flatten()
            Q_11 = ds[f"Q_11_{key}"].isel(integration_time=-1).values.flatten()
            I_02 = ds[f"I_02_{key}"].isel(integration_time=-1).values.flatten()
            Q_02 = ds[f"Q_02_{key}"].isel(integration_time=-1).values.flatten()

            all_I = np.concatenate([I_11, I_02])
            all_Q = np.concatenate([Q_11, Q_02])
            ax.hist2d(all_I / u.mV, all_Q / u.mV, bins=50, cmap="inferno")

            model = _fit_components_for_plot(I_11, Q_11, I_02, Q_02)
            mu_11 = model["mu_11"]
            mu_02_ref = model["mu_ref"]
            used_double_gaussian = bool(model["used_double_gaussian"])

            _draw_gaussian_ellipse_from_stats(
                ax,
                mu_11 / u.mV,
                model["cov_11"] / (u.mV**2),
                color="cyan",
                label="(1,1)",
            )
            ax.plot(*mu_11 / u.mV, "x", color="cyan", ms=10, mew=2)

            # Plot all fitted (0,2) components: one in normal mode, two in T1-limited mode.
            for comp in model["components_02"]:
                is_ref = bool(comp["is_ref"])
                color = "lime" if is_ref else "yellowgreen"
                label = "(0,2)-ref" if is_ref else "(0,2)-aux"
                _draw_gaussian_ellipse_from_stats(
                    ax,
                    np.asarray(comp["mu_iq"]) / u.mV,
                    np.asarray(comp["cov_iq"]) / (u.mV**2),
                    color=color,
                    label=label,
                    lw=2.0 if is_ref else 1.3,
                )
                ax.plot(
                    comp["mu_iq"][0] / u.mV,
                    comp["mu_iq"][1] / u.mV,
                    "x",
                    color=color,
                    ms=9 if is_ref else 7,
                    mew=2 if is_ref else 1.5,
                )

            ax.plot(
                [mu_11[0] / u.mV, mu_02_ref[0] / u.mV],
                [mu_11[1] / u.mV, mu_02_ref[1] / u.mV],
                "w--",
                lw=1.2,
                alpha=0.8,
            )

            t_max = float(ds.integration_time.values[-1])
            ax.set_xlabel("I [mV]")
            ax.set_ylabel("Q [mV]")
            subtitle = f"{key}\nt_int = {t_max:.0f} ns | SNR = {model['snr']:.2f}"
            if used_double_gaussian:
                subtitle += " | (0,2): 2G"
            ax.set_title(subtitle)
            ax.legend(fontsize=7, loc="upper right")
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("IQ Distribution at Longest Integration Time")
    fig.tight_layout()
    return fig


def _draw_gaussian_ellipse_from_stats(ax, mu, cov, color, label, n_std=2, lw=1.5):
    """Draw an ellipse from explicit mean and covariance."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))

    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        lw=lw,
        label=label,
    )
    ax.add_patch(ellipse)


def plot_snr_vs_integration_time(
    ds_fit: xr.Dataset,
    all_sensors: Dict,
    quantum_dot_pairs: List,
    fit_results: Dict = None,
) -> Figure:
    """SNR vs integration time for each sensor/dot-pair."""
    n_cols = sum(len(all_sensors[dp.name]) for dp in quantum_dot_pairs)
    fig, axes = plt.subplots(
        1, max(n_cols, 1), figsize=(5 * max(n_cols, 1), 4), squeeze=False
    )
    axes = axes.flatten()

    col = 0
    for dp in quantum_dot_pairs:
        for sensor in all_sensors[dp.name]:
            key = f"{dp.name}_{sensor.name}"
            ax = axes[col]

            snr = ds_fit[f"snr_{key}"]
            t_int = snr.integration_time.values
            ax.plot(t_int, snr.values, "o-", markersize=4)

            if fit_results is not None and key in fit_results:
                fr = fit_results[key]
                ax.axhline(
                    fr["threshold_snr"], color="r", ls="--", lw=1, alpha=0.6,
                    label=f"SNR threshold = {fr['threshold_snr']:.1f}",
                )
                ax.axvline(
                    fr["optimal_integration_time"], color="g", ls=":", lw=1, alpha=0.6,
                    label=f"Optimal @ {fr['optimal_integration_time']:.0f} ns",
                )
            else:
                ax.axhline(1.0, color="r", ls="--", lw=1, alpha=0.6, label="SNR = 1")

            ax.set_xlabel("Integration time [ns]")
            ax.set_ylabel("SNR")
            ax.set_title(key)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("SNR vs Integration Time")
    fig.tight_layout()
    return fig


def plot_projected_histogram(
    ds: xr.Dataset,
    all_sensors: Dict,
    quantum_dot_pairs: List,
    fit_results: Dict,
) -> Figure:
    """1D histograms projected onto the rotated I axis (integration weight frame) at the optimal
    integration time.  Rotates using the fitted iw_angle so the plot frame matches what the
    hardware will see after the state update."""
    n_cols = sum(len(all_sensors[dp.name]) for dp in quantum_dot_pairs)
    fig, axes = plt.subplots(
        1, max(n_cols, 1), figsize=(6 * max(n_cols, 1), 4), squeeze=False
    )
    axes = axes.flatten()

    col = 0
    integration_times = ds.integration_time.values
    for dp in quantum_dot_pairs:
        for sensor in all_sensors[dp.name]:
            key = f"{dp.name}_{sensor.name}"
            ax = axes[col]
            fr = fit_results[key]

            iw_angle = fr["iw_angle"]
            threshold = fr["I_threshold"]
            cos_a, sin_a = np.cos(iw_angle), np.sin(iw_angle)

            # Use data at the optimal integration time
            opt_t = fr["optimal_integration_time"]
            t_opt_idx = int(np.argmin(np.abs(integration_times - opt_t)))

            I_11 = ds[f"I_11_{key}"].isel(integration_time=t_opt_idx).values.flatten()
            Q_11 = ds[f"Q_11_{key}"].isel(integration_time=t_opt_idx).values.flatten()
            I_02 = ds[f"I_02_{key}"].isel(integration_time=t_opt_idx).values.flatten()
            Q_02 = ds[f"Q_02_{key}"].isel(integration_time=t_opt_idx).values.flatten()

            proj_11 = (I_11 * cos_a + Q_11 * sin_a) / u.mV
            proj_02 = (I_02 * cos_a + Q_02 * sin_a) / u.mV
            threshold_mV = threshold / u.mV

            all_proj = np.concatenate([proj_11, proj_02])
            bins = np.linspace(all_proj.min(), all_proj.max(), 40)

            ax.hist(proj_11, bins=bins, alpha=0.6, color="cyan", label="(1,1)")
            ax.hist(proj_02, bins=bins, alpha=0.6, color="lime", label="(0,2)")
            ax.axvline(
                threshold_mV, color="red", ls="--", lw=1.5,
                label=f"threshold = {threshold_mV:.2f} mV",
            )

            title = f"{key} | t_opt = {opt_t:.0f} ns"
            if fr.get("used_double_gaussian"):
                title += "\n[double Gaussian: T1-limited (0,2)]"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Rotated I [mV]")
            ax.set_ylabel("Counts")
            ax.legend(fontsize=7)
            col += 1

    for idx in range(col, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Projected IQ Distribution (Rotated I Frame, Optimal Integration Time)")
    fig.tight_layout()
    return fig
