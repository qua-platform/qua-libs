"""Figures for fixed-point PSB readout (06d): labeled IQ blobs, histograms, and model fits."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm as _scipy_norm
import xarray as xr


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def plot_single_histogram_with_fit(
    ds_raw: Any,
    ds_fit: xr.Dataset,
    qubit_pairs: Sequence[Union[str, Any]],
    *,
    sweep_name: str = "singleton",
    n_bins: int = 120,
) -> plt.Figure:
    """One subplot per pair: unnormalized I_rot histogram + Barthel analytic densities + I_threshold.

    ``ds_fit`` must contain ``irot_scale`` and ``irot_offset`` (from
    :func:`~calibration_utils.iq_blobs.fit_barthel_mixed_iq`) so that normalized PCA
    quantities can be mapped to the physical rotated-I axis used by state updates.

    Falls back to normalized PCA space when ``irot_scale``/``irot_offset`` are absent.
    """
    names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]
    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
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

        ax.hist(y_plot, bins=n_bins, alpha=0.45, density=True, label="Data (histogram)", color="0.45")
        ax.plot(xs_plot, total, lw=2.0, label="Total (analytic)", color="black")
        ax.plot(xs_plot, S_comp, ls="--", label="S component", color="C0")
        ax.plot(xs_plot, T_no_comp, ls="--", label="T (no decay)", color="C2")
        ax.plot(xs_plot, T_dec_comp, ls="--", label="T (decay)", color="C1")

        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.4g}")

        w = np.asarray(fit.weights.values, dtype=float).ravel()
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

    fig.suptitle("PSB readout: Barthel model (histogram + fit + threshold)")
    fig.tight_layout()
    return fig


def _plot_iq_kde(ax: plt.Axes, I: np.ndarray, Q: np.ndarray, n_grid: int = 100):
    """Render a 2-D KDE density on *ax* and return the grid extents."""
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


def plot_rotated_iq_density(
    ds_raw: Any,
    fit_results: Dict[str, Any],
    qubit_pairs: Sequence[Union[str, Any]],
    *,
    n_grid: int = 100,
) -> plt.Figure:
    """Two subplots per qubit pair: raw IQ density (left) and rotated IQ density with threshold (right).

    The right panel rotates the raw shot-by-shot ``I``, ``Q`` by ``iw_angle`` from *fit_results*
    so that the singlet/triplet separation axis aligns with ``I_rot``, and overlays the
    ``I_threshold`` used by the state update.

    Parameters
    ----------
    ds_raw:
        Dataset with variables ``I`` and ``Q``, dims ``(qubit_pair, n_runs)``.
    fit_results:
        Dict ``{pair_name: {"iw_angle": float, "I_threshold": float, ...}}``.
    qubit_pairs:
        Sequence of qubit-pair objects or name strings (for subplot titles and ``sel``).
    n_grid:
        Resolution of the KDE evaluation grid in each axis.
    """
    names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]
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

        _plot_iq_kde(ax_raw, I_raw, Q_raw, n_grid=n_grid)
        ax_raw.set_xlabel("I")
        ax_raw.set_ylabel("Q")
        ax_raw.set_title(f"{name}  (raw)")

        cos_a, sin_a = np.cos(iw_angle), np.sin(iw_angle)
        I_rot = I_raw * cos_a + Q_raw * sin_a
        Q_rot = -I_raw * sin_a + Q_raw * cos_a

        _plot_iq_kde(ax_rot, I_rot, Q_rot, n_grid=n_grid)
        ax_rot.axvline(I_threshold, color="r", ls="--", lw=1.5, label=f"Threshold = {I_threshold:.4g}")
        ax_rot.set_xlabel("I (rotated)")
        ax_rot.set_ylabel("Q (rotated)")
        ax_rot.set_title(f"{name}  (rotated by iw_angle)")
        ax_rot.legend(loc="upper right", fontsize=8)

    fig.suptitle("PSB readout: IQ density (raw + rotated) + threshold")
    fig.tight_layout()
    return fig


def plot_rotated_iq_density_at_optimum(
    ds_raw: Any,
    fit_results: Dict[str, Any],
    qubit_pairs: Sequence[Union[str, Any]],
    *,
    n_grid: int = 100,
) -> plt.Figure:
    """One subplot per qubit pair: rotated IQ density at the optimal sweep point.

    For each pair, reads ``optimal_sweep_index`` and ``sweep_name`` from *fit_results*
    to slice the shot distribution at the best measured detuning / readout-length /
    ramp-duration, then applies the same KDE + threshold visualisation as
    :func:`plot_rotated_iq_density`.

    ``ds_raw`` must have ``I`` and ``Q`` with dims
    ``(qubit_pair, n_runs, <sweep_dim>)`` where ``<sweep_dim>`` matches
    ``fit_results[name]["sweep_name"]``.
    """
    names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]
    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)
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

        i_min, i_max = I_rot.min(), I_rot.max()
        q_min, q_max = Q_rot.min(), Q_rot.max()
        i_pad = 0.1 * (i_max - i_min) or 1e-9
        q_pad = 0.1 * (q_max - q_min) or 1e-9
        i_grid = np.linspace(i_min - i_pad, i_max + i_pad, n_grid)
        q_grid = np.linspace(q_min - q_pad, q_max + q_pad, n_grid)
        II, QQ = np.meshgrid(i_grid, q_grid)
        kde = gaussian_kde(np.vstack([I_rot, Q_rot]))
        density = kde(np.vstack([II.ravel(), QQ.ravel()])).reshape(n_grid, n_grid)

        ax.imshow(
            density,
            origin="lower",
            aspect="auto",
            extent=[i_grid[0], i_grid[-1], q_grid[0], q_grid[-1]],
            cmap="viridis",
        )
        ax.axvline(I_threshold, color="r", ls="--", lw=1.5, label=f"Threshold = {I_threshold:.4g}")
        ax.set_xlabel("I (rotated)")
        ax.set_ylabel("Q (rotated)")
        ax.set_title(f"{name}  [{sweep_dim}={result.get('optimal_sweep_value', opt_idx):.4g}]")
        ax.legend(loc="upper right", fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("PSB readout at optimum: rotated IQ density + threshold")
    fig.tight_layout()
    return fig


def plot_labeled_histogram_barthel(
    ds_labeled: xr.Dataset,
    ds_fit: xr.Dataset,
    qubits: Sequence[Any],
    *,
    n_bins: int = 80,
) -> plt.Figure:
    """Per-qubit 1D histograms of labeled S/T shots (normalized PCA axis) + Barthel analytic densities.

    ``ds_fit`` must come from :func:`~calibration_utils.iq_blobs.fit_raw_data` and contain
    ``y_pca`` (S then T shots concatenated), ``density_grid``, ``density_S``, ``density_T_no``,
    ``density_T_dec``, ``density_total``, ``norm_ge_threshold``, and ``weights``.

    The first half of ``y_pca`` along ``n_samples`` is S shots; the second half is T shots —
    matching the stack order used by ``fit_raw_data`` (X_ground then X_excited).
    """
    qnames = [q.name for q in qubits]
    n = len(qnames)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, qname in enumerate(qnames):
        ax = axes_flat[idx]
        fit = ds_fit.sel(qubit=qname)

        n_runs = int(ds_labeled["Ig"].sel(qubit=qname).shape[-1])
        y_all = np.asarray(fit.y_pca.values, dtype=float).ravel()
        y_s = y_all[:n_runs]
        y_t = y_all[n_runs: n_runs * 2]

        xs = np.asarray(fit.density_grid.values, dtype=float)
        total    = np.asarray(fit.density_total.values,  dtype=float)
        S_comp   = np.asarray(fit.density_S.values,      dtype=float)
        T_no     = np.asarray(fit.density_T_no.values,   dtype=float)
        T_dec    = np.asarray(fit.density_T_dec.values,  dtype=float)

        ax.hist(y_s[np.isfinite(y_s)], bins=n_bins, alpha=0.35, density=True,
                label="S shots", color="C0")
        ax.hist(y_t[np.isfinite(y_t)], bins=n_bins, alpha=0.35, density=True,
                label="T shots", color="C1")
        ax.plot(xs, total,  lw=2.0, color="black", label="Total (Barthel)")
        ax.plot(xs, S_comp, lw=1.2, ls="--", color="C0",  label="S component")
        ax.plot(xs, T_no,   lw=1.2, ls="--", color="C2",  label="T (no decay)")
        ax.plot(xs, T_dec,  lw=1.2, ls="--", color="C3",  label="T (decay)")

        thr = float(np.asarray(fit.norm_ge_threshold.values).ravel()[0])
        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.3g}")

        w = np.asarray(fit.weights.values, dtype=float).ravel()
        if w.size >= 3:
            ax.text(0.02, 0.98,
                    f"w_S={w[0]:.2f}  w_T(no)={w[1]:.2f}  w_T(dec)={w[2]:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8)

        fid = float(np.asarray(fit.fidelity_opt.values).ravel()[0]) * 100
        ax.set_title(f"{qname}  (F = {fid:.1f} %)")
        ax.set_xlabel("PCA readout (normalized)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("PSB readout: Barthel model — labeled S/T histograms + analytic fit")
    fig.tight_layout()
    return fig


def plot_labeled_histogram_gmm(
    ds_gmm_fit: xr.Dataset,
    qubits: Sequence[Any],
    *,
    n_bins: int = 80,
) -> plt.Figure:
    """Per-qubit 1D histograms of labeled S/T shots (raw PCA axis) + GMM Gaussian components.

    ``ds_gmm_fit`` must be the dataset returned alongside ``fit_results`` by
    ``_fit_gmm_labeled`` and contain:
    ``y_g``, ``y_e``          — S and T shots in raw (un-normalized) PCA space.
    ``gmm_mean_S/T``,
    ``gmm_std_S/T``,
    ``gmm_weight_S/T``        — fitted GMM component parameters.
    ``ge_threshold``           — optimal threshold in the same space.
    ``readout_fidelity``       — analytic fidelity at the optimal threshold (0–1 scale).
    """
    qnames = [q.name for q in qubits]
    n = len(qnames)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, qname in enumerate(qnames):
        ax = axes_flat[idx]
        fit = ds_gmm_fit.sel(qubit=qname)

        y_g = np.asarray(fit.y_g.values, dtype=float).ravel()
        y_e = np.asarray(fit.y_e.values, dtype=float).ravel()
        y_g = y_g[np.isfinite(y_g)]
        y_e = y_e[np.isfinite(y_e)]

        m_S, s_S, w_S = (float(fit.gmm_mean_S),   float(fit.gmm_std_S),   float(fit.gmm_weight_S))
        m_T, s_T, w_T = (float(fit.gmm_mean_T),   float(fit.gmm_std_T),   float(fit.gmm_weight_T))
        thr             = float(fit.ge_threshold)
        fid             = float(fit.readout_fidelity) * 100.0

        lo = min(m_S - 4 * s_S, m_T - 4 * s_T)
        hi = max(m_S + 4 * s_S, m_T + 4 * s_T)
        xs = np.linspace(lo, hi, 600)

        # Bin edges over the same [lo, hi] window as the PDF curves so that bin width
        # matches the distribution width and histogram density aligns with the PDF scale.
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        ax.hist(y_g, bins=bin_edges, alpha=0.35, density=True, label="S shots",  color="C0")
        ax.hist(y_e, bins=bin_edges, alpha=0.35, density=True, label="T shots",  color="C1")
        ax.plot(xs, w_S * _scipy_norm.pdf(xs, m_S, s_S),
                lw=1.2, ls="--", color="C0", label="S component")
        ax.plot(xs, w_T * _scipy_norm.pdf(xs, m_T, s_T),
                lw=1.2, ls="--", color="C1", label="T component")
        ax.plot(xs, w_S * _scipy_norm.pdf(xs, m_S, s_S) + w_T * _scipy_norm.pdf(xs, m_T, s_T),
                lw=2.0, color="black", label="Total (GMM)")

        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.3g}")

        ax.set_title(f"{qname}  (F = {fid:.1f} %)")
        ax.set_xlabel("PCA readout")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("PSB readout: GMM — labeled S/T histograms + Gaussian components")
    fig.tight_layout()
    return fig
