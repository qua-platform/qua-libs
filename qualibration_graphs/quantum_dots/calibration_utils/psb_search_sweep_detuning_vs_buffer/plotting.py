from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr


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
        fig.colorbar(im, ax=ax, label=metric.attrs.get("long_name", metric_name))
        ax.set_title(f"{pair_name} - {metric_name}")
        ax.set_xlabel("Detuning (V)")
        ax.set_ylabel("Buffer duration (ns)")

    fig.suptitle("PSB sweep: detuning vs buffer duration")
    fig.tight_layout()
    return fig
