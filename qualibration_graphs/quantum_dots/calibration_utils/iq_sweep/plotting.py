from typing import List, Any, Optional
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _grid(n: int):
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _sweep_axis_label(fits: xr.Dataset, sweep_name: str) -> str:
    attrs = fits[sweep_name].attrs if sweep_name in fits.coords else {}
    long_name = attrs.get("long_name", sweep_name)
    units = attrs.get("units")
    return f"{long_name} [{units}]" if units else long_name


def plot_metric_vs_sweep(
    ds: xr.Dataset,
    qubit_pairs: List[Any],
    fits: xr.Dataset,
    sweep_name: str,
    metric_field: str,
    metric_label: str,
    optimum_field: Optional[str] = None,
) -> Figure:
    """Per-qubit subplot of a scalar metric vs the sweep coordinate.

    If ``optimum_field`` is given and present in ``fits``, draws a vertical
    marker at that optimum value per qubit.
    """
    n = len(qubit_pairs)
    n_rows, n_cols = _grid(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    sweep_vals = np.asarray(fits[sweep_name].values)
    x_label = _sweep_axis_label(fits, sweep_name)

    for idx, qp in enumerate(qubit_pairs):
        ax = axes_flat[idx]
        fit_q = fits.sel(qubit_pair=qp.name)
        y = np.asarray(fit_q[metric_field].values)
        ax.plot(sweep_vals, y, "-o", ms=4)

        if optimum_field is not None and optimum_field in fits:
            opt = float(fit_q[optimum_field].values)
            if np.isfinite(opt):
                ax.axvline(opt, color="r", ls="--", lw=1.5, label=f"opt = {opt:.4g}")
                ax.legend(fontsize=8)

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(qp.name)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"{metric_label} vs {sweep_name}")
    fig.tight_layout()
    return fig


def plot_fidelity_vs_sweep(
    ds: xr.Dataset,
    qubit_pairs: List[Any],
    fits: xr.Dataset,
    sweep_name: str = "detuning",
) -> Figure:
    return plot_metric_vs_sweep(
        ds,
        qubit_pairs,
        fits,
        sweep_name=sweep_name,
        metric_field="readout_fidelity",
        metric_label="Readout fidelity (%)",
        optimum_field="optimal_sweep_value_fidelity",
    )


def plot_visibility_vs_sweep(
    ds: xr.Dataset,
    qubit_pairs: List[Any],
    fits: xr.Dataset,
    sweep_name: str = "detuning",
) -> Figure:
    return plot_metric_vs_sweep(
        ds,
        qubit_pairs,
        fits,
        sweep_name=sweep_name,
        metric_field="visibility_opt",
        metric_label="Visibility",
        optimum_field="optimal_sweep_value_visibility",
    )


def plot_sweep_summary(
    ds: xr.Dataset,
    qubit_pairs: List[Any],
    fits: xr.Dataset,
    sweep_name: str = "detuning",
) -> Figure:
    """Fidelity and visibility on twin y-axes per qubit pair with both optima marked."""
    n = len(qubit_pairs)
    n_rows, n_cols = _grid(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    sweep_vals = np.asarray(fits[sweep_name].values)
    x_label = _sweep_axis_label(fits, sweep_name)

    for idx, qp in enumerate(qubit_pairs):
        ax = axes_flat[idx]
        fit_q = fits.sel(qubit_pair=qp.name)

        ax.plot(
            sweep_vals,
            np.asarray(fit_q.readout_fidelity.values),
            "-o",
            color="C0",
            ms=4,
            label="Fidelity (%)",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Fidelity (%)", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")

        ax2 = ax.twinx()
        ax2.plot(
            sweep_vals,
            np.asarray(fit_q.visibility_opt.values),
            "-s",
            color="C1",
            ms=4,
            label="Visibility",
        )
        ax2.set_ylabel("Visibility", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        opt_f = float(fit_q.optimal_sweep_value_fidelity.values)
        opt_v = float(fit_q.optimal_sweep_value_visibility.values)
        title_bits = [qp.name]
        if np.isfinite(opt_f):
            ax.axvline(opt_f, color="C0", ls="--", lw=1, alpha=0.7)
            title_bits.append(f"F* @ {opt_f:.4g}")
        if np.isfinite(opt_v):
            ax.axvline(opt_v, color="C1", ls=":", lw=1, alpha=0.7)
            title_bits.append(f"V* @ {opt_v:.4g}")

        ax.set_title("  |  ".join(title_bits))
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Fidelity & Visibility vs {sweep_name}")
    fig.tight_layout()
    return fig
