from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr

__all__ = ["plot_detuning_vs_buffer_pca_map", "plot_all"]


def plot_detuning_vs_buffer_pca_map(
    ds_fit: xr.Dataset,
    *,
    metric_name: str = "pc1_std",
) -> plt.Figure:
    """Plot detuning-vs-buffer heatmaps of a PCA-derived contrast metric."""
    pair_names = list(ds_fit["qubit_pair"].values)
    n_pairs = max(len(pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)

    for idx, pair_name in enumerate(pair_names):
        ax = axes[0, idx]
        metric = ds_fit[metric_name].sel(qubit_pair=pair_name)
        detuning = ds_fit["detuning"].values
        buffer_duration = ds_fit["buffer_duration"].values

        im = ax.pcolormesh(
            detuning,
            buffer_duration,
            metric.values.T,
            shading="nearest",
            cmap="viridis",
        )
        metric_label = metric.attrs.get("long_name", metric_name)
        fig.colorbar(im, ax=ax, label=metric_label)
        ax.set_title(f"{pair_name} - {metric_label}")
        ax.set_xlabel("Detuning (V)")
        ax.set_ylabel("Buffer duration (ns)")

    fig.suptitle(f"PSB sweep: {metric_label} vs detuning and buffer duration")
    fig.tight_layout()
    return fig


def plot_all(ds_fit: xr.Dataset, *, metric_name: str = "pc1_std") -> dict[str, plt.Figure]:
    """Generate all node figures via the local plotting API."""
    # 06e currently exposes a single summary heatmap. Keeping this wrapper means
    # the node can stay consistent with the other PSB nodes even though the
    # plotting stack here is intentionally much lighter than 06a-06d.
    fig = plot_detuning_vs_buffer_pca_map(ds_fit, metric_name=metric_name)
    return {"detuning_vs_buffer_pca_map": fig}
