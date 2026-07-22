import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_heralded_n_loops",
    "plot_heralded_n_loops_2d"
]


def plot_heralded_n_loops(
    ds_raw,
    item_names: list[str],
    *,
    item_dim: str,
    sweep_key: str,
    sweep_scale: float = 1.0,
    sweep_xlabel: str = "",
):
    """Plot average heralded loop count vs a sweep axis for each item."""

    if not item_names:
        return None

    fig, axes = plt.subplots(1, len(item_names), figsize=(7 * len(item_names), 5), squeeze=False)
    sweep_vals = ds_raw[sweep_key].values / sweep_scale

    for idx, item_name in enumerate(item_names):
        ax = axes[0, idx]
        candidates = [
            f"n_loops_{item_name}",
            f"n_loops_{item_name.rstrip('0123456789')}",
        ]
        n_key = next((key for key in candidates if key in ds_raw), None)
        if n_key is None:
            continue

        n_loops_vals = ds_raw[n_key].sel({item_dim: item_name}).values
        ax.plot(sweep_vals, n_loops_vals, color="C2", linestyle="--", label="n_loops")
        ax.set_xlabel(sweep_xlabel)
        ax.set_ylabel("n_loops")
        ax.set_title(f"n_loops vs {sweep_key} - {item_name}")
        ax.legend()

    fig.suptitle("Heralded initialization loop count")
    fig.tight_layout()
    return fig


def plot_heralded_n_loops_2d(
    ds_raw,
    item_names: list[str],
    *,
    item_dim: str,
    x_sweep_key: str,
    y_sweep_key: str,
    x_sweep_scale: float = 1.0,
    y_sweep_scale: float = 1.0,
    x_sweep_xlabel: str = "",
    y_sweep_ylabel: str = "",
):
    """Plot average heralded loop count on a 2-D sweep for each item."""

    if not item_names:
        return None

    fig, axes = plt.subplots(1, len(item_names), figsize=(7 * len(item_names), 5), squeeze=False)
    x_vals = ds_raw[x_sweep_key].values / x_sweep_scale
    y_vals = ds_raw[y_sweep_key].values / y_sweep_scale

    for idx, item_name in enumerate(item_names):
        ax = axes[0, idx]
        candidates = [
            f"n_loops_{item_name}",
            f"n_loops_{item_name.rstrip('0123456789')}",
        ]
        n_key = next((key for key in candidates if key in ds_raw), None)
        if n_key is None:
            continue

        n_loops_da = ds_raw[n_key].sel({item_dim: item_name})
        if (
            hasattr(n_loops_da, "dims")
            and y_sweep_key in n_loops_da.dims
            and x_sweep_key in n_loops_da.dims
        ):
            n_loops_da = n_loops_da.transpose(y_sweep_key, x_sweep_key)
        n_loops_vals = np.asarray(n_loops_da.values, dtype=float)
        image = ax.pcolormesh(x_vals, y_vals, n_loops_vals, shading="auto", cmap="viridis")
        ax.set_xlabel(x_sweep_xlabel)
        ax.set_ylabel(y_sweep_ylabel)
        ax.set_title(f"n_loops map - {item_name}")
        fig.colorbar(image, ax=ax, label="n_loops")

    fig.suptitle("Heralded initialization loop count")
    fig.tight_layout()
    return fig
