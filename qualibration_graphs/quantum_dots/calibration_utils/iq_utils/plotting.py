"""Shared plotting utilities for IQ readout calibration.

This module contains plotting primitives that are re-used by both fixed-point
and sweep-style IQ readout calibrations.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde


def _grid(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _sweep_axis_label(fits: xr.Dataset, sweep_name: str) -> str:
    attrs = fits[sweep_name].attrs if sweep_name in fits.coords else {}
    long_name = attrs.get("long_name", sweep_name)
    units = attrs.get("units")
    return f"{long_name} [{units}]" if units else long_name


def _grid_subplots(n: int) -> tuple[int, int]:
    return _grid(n)


def _plot_iq_kde(ax: plt.Axes, I: np.ndarray, Q: np.ndarray, *, n_grid: int = 100) -> None:
    """Render a 2-D KDE density on *ax*."""
    i_min, i_max = I.min(), I.max()
    q_min, q_max = Q.min(), Q.max()
    i_pad = 0.1 * (i_max - i_min) or 1e-9
    q_pad = 0.1 * (q_max - q_min) or 1e-9
    i_grid = np.linspace(i_min - i_pad, i_max + i_pad, n_grid)
    q_grid = np.linspace(q_min - q_pad, q_max + q_pad, n_grid)
    II, QQ = np.meshgrid(i_grid, q_grid)
    kde = gaussian_kde(np.vstack([I, Q]))
    density = kde(np.vstack([II.ravel(), QQ.ravel()])).reshape(n_grid, n_grid)
    ax.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=[i_grid[0], i_grid[-1], q_grid[0], q_grid[-1]],
        cmap="viridis",
    )

def _plot_iq_scatter(
    ax: plt.Axes, I: np.ndarray, Q: np.ndarray, *, s: float = 4, alpha: float = 0.15
) -> None:
    """Render raw I/Q shots as a scatter (density emerges from point overlap)."""
    ax.scatter(I, Q, s=s, alpha=alpha, edgecolors="none", color="C0", rasterized=True)
    ax.set_aspect("auto")


def _names(items: Sequence[Union[str, Any]]) -> list[str]:
    return [x if isinstance(x, str) else getattr(x, "name", str(x)) for x in items]


def plot_rotated_iq_density(
    ds_raw: Any,
    fit_results: dict[str, Any],
    items: Sequence[Union[str, Any]],
    *,
    n_grid: int = 100,
    plot_kde: bool = True,
    alpha: float = 0.15, 
    s: float = 4,
) -> Figure:
    """Two subplots per item: raw IQ density and rotated IQ density + threshold."""
    names = _names(items)
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.5 * n), squeeze=False)

    for idx, name in enumerate(names):
        ax_raw, ax_rot = axes[idx, 0], axes[idx, 1]

        if name not in fit_results:
            ax_raw.set_title(f"{name} (no fit result)")
            ax_rot.set_title(f"{name} (no fit result)")
            continue

        result = fit_results[name]
        iw_angle = float(result["iw_angle"])
        I_threshold = float(result["I_threshold"])

        I_raw = np.asarray(ds_raw.I.sel(qubit_pair=name).values, dtype=float).ravel()
        Q_raw = np.asarray(ds_raw.Q.sel(qubit_pair=name).values, dtype=float).ravel()
        finite = np.isfinite(I_raw) & np.isfinite(Q_raw)
        I_raw, Q_raw = I_raw[finite], Q_raw[finite]

        if I_raw.size < 4:
            ax_raw.set_title(f"{name} (insufficient data)")
            ax_rot.set_title(f"{name} (insufficient data)")
            continue

        if plot_kde: 
            _plot_iq_kde(ax_raw, I_raw, Q_raw, n_grid=n_grid)
        else: 
            _plot_iq_scatter(ax_raw, I_raw, Q_raw, alpha = alpha, s = s)
        ax_raw.set_xlabel("I")
        ax_raw.set_ylabel("Q")
        ax_raw.set_title(f"{name}  (raw)")

        cos_a, sin_a = np.cos(iw_angle), np.sin(iw_angle)
        I_rot = I_raw * cos_a + Q_raw * sin_a
        Q_rot = -I_raw * sin_a + Q_raw * cos_a

        if plot_kde: 
            _plot_iq_kde(ax_rot, I_rot, Q_rot, n_grid=n_grid)
        else: 
            _plot_iq_scatter(ax_rot, I_rot, Q_rot, alpha = alpha, s = s)
        ax_rot.axvline(
            I_threshold,
            color="r",
            ls="--",
            lw=1.5,
            label=f"Threshold = {I_threshold:.4g}",
        )
        ax_rot.set_xlabel("I (rotated)")
        ax_rot.set_ylabel("Q (rotated)")
        ax_rot.set_title(f"{name}  (rotated by iw_angle)")
        ax_rot.legend(loc="upper right", fontsize=8)

    fig.suptitle("Readout: IQ density (raw + rotated) + threshold")
    fig.tight_layout()
    return fig


def plot_rotated_iq_density_at_optimum(
    ds_raw: Any,
    fit_results: dict[str, Any],
    items: Sequence[Union[str, Any]],
    *,
    n_grid: int = 100,
    plot_kde: bool = True,
    alpha: float = 0.15, 
    s: float = 4,
) -> Figure:
    """One subplot per item: rotated IQ density at the optimal sweep point."""
    names = _names(items)
    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]

        if name not in fit_results:
            ax.set_title(f"{name} (no fit result)")
            continue

        result = fit_results[name]
        iw_angle = float(result["iw_angle"])
        I_threshold = float(result["I_threshold"])
        opt_idx = int(result["optimal_sweep_index"])
        sweep_dim = str(result.get("sweep_name", ""))

        if not np.isfinite(iw_angle):
            ax.set_title(f"{name} (iw_angle not available)")
            continue

        try:
            I_raw = np.asarray(
                ds_raw.I.sel(qubit_pair=name).isel({sweep_dim: opt_idx}).values, dtype=float
            ).ravel()
            Q_raw = np.asarray(
                ds_raw.Q.sel(qubit_pair=name).isel({sweep_dim: opt_idx}).values, dtype=float
            ).ravel()
        except Exception as exc:
            ax.set_title(f"{name} (data error: {exc})")
            continue

        finite = np.isfinite(I_raw) & np.isfinite(Q_raw)
        I_raw, Q_raw = I_raw[finite], Q_raw[finite]
        if I_raw.size < 4:
            ax.set_title(f"{name} (insufficient data)")
            continue

        cos_a, sin_a = np.cos(iw_angle), np.sin(iw_angle)
        I_rot = I_raw * cos_a + Q_raw * sin_a
        Q_rot = -I_raw * sin_a + Q_raw * cos_a

        if plot_kde: 
            _plot_iq_kde(ax, I_rot, Q_rot, n_grid = n_grid)
        else: 
            _plot_iq_scatter(ax, I_rot, Q_rot, alpha = alpha, s = s)

        ax.axvline(
            I_threshold,
            color="r",
            ls="--",
            lw=1.5,
            label=f"Threshold = {I_threshold:.4g}",
        )
        ax.set_xlabel("I (rotated)")
        ax.set_ylabel("Q (rotated)")
        ax.set_title(
            f"{name}  [{sweep_dim}={result.get('optimal_sweep_value', opt_idx):.4g}]"
        )
        ax.legend(loc="upper right", fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Readout at optimum: rotated IQ density + threshold")
    fig.tight_layout()
    return fig


def plot_single_histogram_with_fit(
    ds_fit: xr.Dataset,
    items: Sequence[Union[str, Any]],
    *,
    n_bins: int = 120,
) -> Figure:
    """One subplot per item: histogram + analytic density components + threshold."""
    names = _names(items)
    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    has_irot = "irot_scale" in ds_fit and "irot_offset" in ds_fit

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        fit = ds_fit.sel(qubit_pair=name)

        y_pca = np.asarray(fit.y_pca.values, dtype=float).ravel()
        y_pca = y_pca[np.isfinite(y_pca)]
        if y_pca.size == 0:
            ax.set_title(f"{name} (no finite y_pca)")
            continue

        xs_norm = np.asarray(fit.density_grid.values, dtype=float)
        total = np.asarray(fit.density_total.values, dtype=float)
        S_comp = np.asarray(fit.density_S.values, dtype=float)
        T_no_comp = np.asarray(fit.density_T_no.values, dtype=float)
        T_dec_comp = np.asarray(fit.density_T_dec.values, dtype=float)

        if has_irot:
            scale = float(fit.irot_scale.values)
            offset = float(fit.irot_offset.values)
            y_plot = y_pca * scale + offset
            xs_plot = xs_norm * scale + offset
            total = total / scale
            S_comp = S_comp / scale
            T_no_comp = T_no_comp / scale
            T_dec_comp = T_dec_comp / scale
            thr = float(np.asarray(fit.I_threshold.values).ravel()[0])
            xlabel = "I (rotated)"
        else:
            y_plot = y_pca
            xs_plot = xs_norm
            thr = float(np.asarray(fit.norm_ge_threshold.values).ravel()[0])
            xlabel = "PCA readout (normalized)"

        ax.hist(
            y_plot,
            bins=n_bins,
            alpha=0.45,
            density=True,
            label="Data (histogram)",
            color="0.45",
        )
        ax.plot(xs_plot, total, lw=2.0, label="Total (analytic)", color="black")
        ax.plot(xs_plot, S_comp, ls="--", label="S component", color="C0")
        ax.plot(xs_plot, T_no_comp, ls="--", label="T (no decay)", color="C2")
        ax.plot(xs_plot, T_dec_comp, ls="--", label="T (decay)", color="C1")

        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.4g}")

        w = np.asarray(fit.weights.values, dtype=float).ravel() if "weights" in ds_fit else np.array([])
        if w.size >= 3:
            ax.text(
                0.02,
                0.98,
                f"w_S={w[0]:.2f}  w_T(no)={w[1]:.2f}  w_T(dec)={w[2]:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Readout: histogram + analytic fit")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_rotated_iq_density",
    "plot_rotated_iq_density_at_optimum",
    "plot_single_histogram_with_fit",
]
