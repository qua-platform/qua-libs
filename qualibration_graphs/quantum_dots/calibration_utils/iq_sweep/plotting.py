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
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
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
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
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


def plot_histograms_vs_sweep(
    ds: xr.Dataset,
    qubit_pairs: List[Any],
    fits: xr.Dataset,
    sweep_name: str = "detuning",
    *,
    n_bins: int = 256,
    log_counts: bool = False,
    normalize_by_sweep: bool = False,
) -> Figure:
    """Per qubit pair: 2D map of shot histograms along the sweep coordinate.

    For each sweep value, shots are projected with that slice's ``iw_angle`` from
    ``fits`` (same convention as the Barthel readout axis), then a 1D histogram
    is accumulated. The vertical axis is the projected readout coordinate; the
    horizontal axis is the sweep. ``I_threshold`` and Gaussian means are overlaid
    when available.

    Parameters
    ----------
    normalize_by_sweep:
        When True, divide each slice's projected values by the corresponding
        sweep coordinate value (e.g. readout length in ns) before histogramming.
        This removes the linear amplitude growth due to ``measure_accumulated``
        cumulative summation, making blobs at all sweep values appear at the
        same vertical position and visually narrowing as SNR improves.
        Use for readout-length sweeps (06b); leave False for detuning sweeps (06a).
    """
    n = len(qubit_pairs)
    n_rows, n_cols = _grid(n)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    sweep_vals = np.asarray(ds[sweep_name].values)
    x_label = _sweep_axis_label(ds, sweep_name)

    for idx, qp in enumerate(qubit_pairs):
        ax = axes_flat[idx]
        name = qp.name
        fit_q = fits.sel(qubit_pair=name)

        # Determine projection mode: use amplitude (I²+Q²) when iw_angle is absent
        # or all-NaN, matching how mu1/mu2 are computed.
        iw_angles = np.asarray(fit_q.iw_angle.values, dtype=float) if "iw_angle" in fits else np.array([np.nan])
        use_amplitude = not np.any(np.isfinite(iw_angles))

        # Build per-sweep projected samples
        proj_cols = []
        for si, v in enumerate(sweep_vals):
            I = np.asarray(ds.I.sel(qubit_pair=name, **{sweep_name: v}).values).ravel()
            Q = np.asarray(ds.Q.sel(qubit_pair=name, **{sweep_name: v}).values).ravel()
            if use_amplitude:
                proj = I ** 2 + Q ** 2
            else:
                ang = float(np.asarray(fit_q.iw_angle.sel(**{sweep_name: v}).values))
                if not np.isfinite(ang):
                    ang = 0.0
                c, s = np.cos(ang), np.sin(ang)
                proj = I * c + Q * s
            if normalize_by_sweep and float(v) != 0.0:
                proj = proj / float(v)
            proj_cols.append(proj)

        flat = np.concatenate(proj_cols) if proj_cols else np.array([0.0])
        lo, hi = np.nanpercentile(flat, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        y_min, y_max = lo - pad, hi + pad

        n_sweep = len(sweep_vals)
        H = np.zeros((n_sweep, n_bins), dtype=float)
        edges = np.linspace(y_min, y_max, n_bins + 1)
        for si in range(n_sweep):
            H[si, :], _ = np.histogram(proj_cols[si], bins=edges, density=False)

        Z = np.log10(H.T + 1.0) if log_counts else H.T
        if log_counts:
            vmin, vmax = np.nanpercentile(Z, [5.0, 99.5])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None

        if n_sweep == 1:
            dv = 1e-6 * (abs(float(sweep_vals[0])) + 1.0)
            x_left = float(sweep_vals[0]) - 0.5 * dv
            x_right = float(sweep_vals[0]) + 0.5 * dv
        else:
            x_left = float(sweep_vals[0])
            x_right = float(sweep_vals[-1])

        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=[x_left, x_right, y_min, y_max],
            interpolation="nearest",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        # Overlay threshold and Gaussian means, normalising to match the plot units
        sweep_safe = np.where(np.abs(sweep_vals) > 0.0, sweep_vals, np.nan)
        thr = np.asarray(fit_q.I_threshold.values, dtype=float)
        thr_plot = thr / sweep_safe if normalize_by_sweep else thr
        if np.any(np.isfinite(thr_plot)):
            ax.plot(sweep_vals, thr_plot, color="cyan", lw=1.5, ls="--", label="I_threshold")

        if "mu1" in fits and "mu2" in fits:
            mu1 = np.asarray(fit_q["mu1"].values, dtype=float)
            mu2 = np.asarray(fit_q["mu2"].values, dtype=float)
            mu1_plot = mu1 / sweep_safe if normalize_by_sweep else mu1
            mu2_plot = mu2 / sweep_safe if normalize_by_sweep else mu2
            if np.any(np.isfinite(mu1_plot)):
                ax.plot(sweep_vals, mu1_plot, color="lime", lw=1.5, ls=":", label="mu1")
            if np.any(np.isfinite(mu2_plot)):
                ax.plot(sweep_vals, mu2_plot, color="deepskyblue", lw=1.5, ls=":", label="mu2")

        if use_amplitude:
            y_label = "Amplitude I²+Q²"
        elif normalize_by_sweep:
            y_label = "Projected readout / sweep value [a.u. per unit]"
        else:
            y_label = "Projected readout (accumulated units)"

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(qp.name)
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="log10(count+1)" if log_counts else "counts",
        )

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Shot histograms vs {sweep_name}"
        + (" [normalised by sweep value]" if normalize_by_sweep else "")
    )
    fig.tight_layout()
    return fig
