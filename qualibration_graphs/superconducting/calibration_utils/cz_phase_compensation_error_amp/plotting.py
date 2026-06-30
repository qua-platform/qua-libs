"""Plotting utilities for CZ phase compensation with error amplification."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Per-channel colors: muted markers for data, bold contrasting lines for the fit.
_PANEL_STYLES = {
    "control": {
        "data": {"color": "#3b82f6", "markeredgecolor": "#1e3a8a"},
        "fit": {"color": "#ea580c"},
        "optimum": {"color": "#15803d"},
    },
    "target": {
        "data": {"color": "#ef4444", "markeredgecolor": "#991b1b"},
        "fit": {"color": "#7c3aed"},
        "optimum": {"color": "#15803d"},
    },
}


def plot_raw_data_with_fit(ds_raw: xr.Dataset, qubit_pairs, ds_fit: xr.Dataset = None):
    """Plot mean signal vs frame for control and target with the sinc fit overlay."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(6 * n_pairs, 8), squeeze=False)

    for i, qp in enumerate(qubit_pairs):
        qp_name = qp.name

        if ds_fit is None or not bool(ds_fit.sel(qubit_pair=qp_name).success.values):
            for row in range(2):
                axes[row, i].text(
                    0.5,
                    0.5,
                    "fit failed" if ds_fit else "no fit",
                    ha="center",
                    va="center",
                    transform=axes[row, i].transAxes,
                )
                axes[row, i].set_title(f"{qp_name} - {'control' if row == 0 else 'target'}")
                axes[row, i].set_xlabel("Frame rotation [2π]")
                axes[row, i].set_ylabel("Mean signal")
            continue

        qp_fit = ds_fit.sel(qubit_pair=qp_name)
        frames = ds_raw.sel(qubit_pair=qp_name).frame.values

        for row, label in enumerate(["control", "target"]):
            ax = axes[row, i]
            style = _PANEL_STYLES[label]
            mean_var = f"{label}_mean_vs_frame"
            sinc_var = f"{label}_sinc_fit"
            phase_var = f"fitted_{label}_phase"
            peak_mean_var = f"{label}_mean_at_peak"
            method_var = f"{label}_fit_method"

            if mean_var in qp_fit.data_vars:
                mean_curve = qp_fit[mean_var].values
                fitted_phase = float(qp_fit[phase_var].values)
                peak_mean = float(qp_fit[peak_mean_var].values)
                fit_method = str(qp_fit[method_var].values)

                ax.plot(
                    frames,
                    mean_curve,
                    linestyle="none",
                    marker="o",
                    ms=5,
                    mew=0.8,
                    alpha=0.9,
                    zorder=2,
                    label=r"$\langle signal \rangle_N$",
                    **style["data"],
                )
                if sinc_var in qp_fit.data_vars:
                    fit_vals = qp_fit[sinc_var].values
                    if np.any(np.isfinite(fit_vals)):
                        ax.plot(
                            frames,
                            fit_vals,
                            "-",
                            lw=2.5,
                            zorder=4,
                            label="sinc fit",
                            **style["fit"],
                        )
                ax.axvline(
                    fitted_phase,
                    color=style["optimum"]["color"],
                    ls="--",
                    lw=1.8,
                    alpha=0.95,
                    zorder=3,
                    label=f"phase = {fitted_phase:.4f} ({fit_method})",
                )
                ax.plot(
                    fitted_phase,
                    peak_mean,
                    "*",
                    ms=14,
                    mew=1.2,
                    zorder=5,
                    markerfacecolor="white",
                    markeredgecolor=style["fit"]["color"],
                    label=f"peak mean = {peak_mean:.4f}",
                )
                ax.legend(loc="best", fontsize=9)

            ax.set_title(f"{qp_name} - {label}")
            ax.set_xlabel("Frame rotation [2π]")
            ax.set_ylabel("Mean signal")
            ax.grid(alpha=0.25, color="#94a3b8")

    for j in range(n_pairs, axes.shape[1]):
        axes[0, j].axis("off")
        axes[1, j].axis("off")

    fig.suptitle("CZ phase compensation - error amplification (sinc fit)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
