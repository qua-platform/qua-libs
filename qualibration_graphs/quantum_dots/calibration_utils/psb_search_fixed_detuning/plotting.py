"""Figures for fixed-point PSB readout (06d): labeled IQ blobs and model fits."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _scipy_norm
import xarray as xr


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


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
        y_t = y_all[n_runs : n_runs * 2]

        xs = np.asarray(fit.density_grid.values, dtype=float)
        total = np.asarray(fit.density_total.values, dtype=float)
        S_comp = np.asarray(fit.density_S.values, dtype=float)
        T_no = np.asarray(fit.density_T_no.values, dtype=float)
        T_dec = np.asarray(fit.density_T_dec.values, dtype=float)

        ax.hist(y_s[np.isfinite(y_s)], bins=n_bins, alpha=0.35, density=True, label="S shots", color="C0")
        ax.hist(y_t[np.isfinite(y_t)], bins=n_bins, alpha=0.35, density=True, label="T shots", color="C1")
        ax.plot(xs, total, lw=2.0, color="black", label="Total (Barthel)")
        ax.plot(xs, S_comp, lw=1.2, ls="--", color="C0", label="S component")
        ax.plot(xs, T_no, lw=1.2, ls="--", color="C2", label="T (no decay)")
        ax.plot(xs, T_dec, lw=1.2, ls="--", color="C3", label="T (decay)")

        thr = float(np.asarray(fit.norm_ge_threshold.values).ravel()[0])
        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.3g}")

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

        m_S, s_S, w_S = (float(fit.gmm_mean_S), float(fit.gmm_std_S), float(fit.gmm_weight_S))
        m_T, s_T, w_T = (float(fit.gmm_mean_T), float(fit.gmm_std_T), float(fit.gmm_weight_T))
        thr = float(fit.ge_threshold)
        fid = float(fit.readout_fidelity) * 100.0

        lo = min(m_S - 4 * s_S, m_T - 4 * s_T)
        hi = max(m_S + 4 * s_S, m_T + 4 * s_T)
        xs = np.linspace(lo, hi, 600)

        # Bin edges over the same [lo, hi] window as the PDF curves so that bin width
        # matches the distribution width and histogram density aligns with the PDF scale.
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        ax.hist(y_g, bins=bin_edges, alpha=0.35, density=True, label="S shots", color="C0")
        ax.hist(y_e, bins=bin_edges, alpha=0.35, density=True, label="T shots", color="C1")
        ax.plot(xs, w_S * _scipy_norm.pdf(xs, m_S, s_S), lw=1.2, ls="--", color="C0", label="S component")
        ax.plot(xs, w_T * _scipy_norm.pdf(xs, m_T, s_T), lw=1.2, ls="--", color="C1", label="T component")
        ax.plot(
            xs,
            w_S * _scipy_norm.pdf(xs, m_S, s_S) + w_T * _scipy_norm.pdf(xs, m_T, s_T),
            lw=2.0,
            color="black",
            label="Total (GMM)",
        )

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
