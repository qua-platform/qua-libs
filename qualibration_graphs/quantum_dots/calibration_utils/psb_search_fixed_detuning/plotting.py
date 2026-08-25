"""Figures for fixed-point PSB readout (06d): labeled IQ blobs and model fits."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm as _scipy_norm
import xarray as xr

__all__ = [
    "plot_all",
    "plot_labeled_histogram_barthel",
    "plot_labeled_histogram_gmm",
    "plot_labeled_iq_blobs",
]


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _rotate_iq(I: np.ndarray, Q: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    ca, sa = np.cos(angle), np.sin(angle)
    return I * ca + Q * sa, -I * sa + Q * ca


def _weighted_hist_density(
    values: np.ndarray, bin_edges: np.ndarray, area: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build histogram bars whose total area matches a target mixture weight.

    Matplotlib's ``density=True`` normalizes each histogram independently to unit area.
    For these 06d overlays we instead want the S/T bars to live on the same pooled-density
    axis as the fitted component PDFs, whose integrals are their mixture weights.
    """

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if len(finite_values) == 0:
        return centers, widths, np.zeros_like(centers)

    counts, _ = np.histogram(finite_values, bins=bin_edges)
    density = float(area) * counts / (len(finite_values) * widths)
    return centers, widths, density


def plot_labeled_iq_blobs(
    ds_labeled: xr.Dataset,
    qubits: Sequence[Any],
    fit_results: Dict[str, Dict[str, Any]],
) -> Figure:
    """Raw IQ (left) and rotated IQ with threshold (right) for each qubit."""

    n_qubits = len(qubits)
    fig, axes = plt.subplots(n_qubits, 2, figsize=(11, 4.5 * n_qubits), squeeze=False)

    for idx, qubit in enumerate(qubits):
        # Each row shows the same shots twice:
        #   left  = raw IQ
        #   right = IQ rotated into the state-update readout axis
        ax_raw, ax_rot = axes[idx, 0], axes[idx, 1]
        name = qubit.name
        Ig = np.asarray(ds_labeled["Ig"].sel(qubit=name).values, dtype=float)
        Qg = np.asarray(ds_labeled["Qg"].sel(qubit=name).values, dtype=float)
        Ie = np.asarray(ds_labeled["Ie"].sel(qubit=name).values, dtype=float)
        Qe = np.asarray(ds_labeled["Qe"].sel(qubit=name).values, dtype=float)

        fit_result = fit_results.get(name, {})
        iw_angle = float(fit_result.get("iw_angle", 0.0))
        I_threshold = fit_result.get("I_threshold")

        # Raw IQ view: useful for checking blob separation before applying the
        # fitted readout-axis rotation.
        ax_raw.plot(Ig * 1e3, Qg * 1e3, ".", alpha=0.4, markersize=2, label="S", color="C0")
        ax_raw.plot(Ie * 1e3, Qe * 1e3, ".", alpha=0.4, markersize=2, label="T", color="C1")
        ax_raw.set_xlabel("I [mV]")
        ax_raw.set_ylabel("Q [mV]")
        ax_raw.set_title(f"{name}  (raw)")
        ax_raw.legend(fontsize=7)
        ax_raw.grid(True, alpha=0.3)

        # Rotated IQ view: this is the axis used for thresholding/state update,
        # so the vertical line is the physically applied discrimination threshold.
        Ig_rot, Qg_rot = _rotate_iq(Ig, Qg, iw_angle)
        Ie_rot, Qe_rot = _rotate_iq(Ie, Qe, iw_angle)
        ax_rot.plot(Ig_rot * 1e3, Qg_rot * 1e3, ".", alpha=0.4, markersize=2, label="S", color="C0")
        ax_rot.plot(Ie_rot * 1e3, Qe_rot * 1e3, ".", alpha=0.4, markersize=2, label="T", color="C1")
        if I_threshold is not None and np.isfinite(I_threshold):
            ax_rot.axvline(
                float(I_threshold) * 1e3,
                color="C3",
                ls="--",
                lw=1.5,
                label=f"I_threshold = {float(I_threshold) * 1e3:.2f} mV",
            )
        ax_rot.set_xlabel("I_rot [mV]")
        ax_rot.set_ylabel("Q_rot [mV]")
        ax_rot.set_title(f"{name}  (rotated by iw_angle)")
        ax_rot.legend(fontsize=7)
        ax_rot.grid(True, alpha=0.3)

    fig.suptitle("PSB IQ blobs — raw + rotated (state-update angle & threshold)")
    fig.tight_layout()
    return fig


def plot_labeled_histogram_barthel(
    ds_labeled: xr.Dataset,
    ds_fit: xr.Dataset,
    qubits: Sequence[Any],
    *,
    n_bins: int = 80,
) -> Figure:
    """Per-qubit labeled S/T histograms with the Barthel analytic fit."""

    qnames = [q.name for q in qubits]
    n = len(qnames)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, qname in enumerate(qnames):
        ax = axes_flat[idx]
        fit = ds_fit.sel(qubit=qname)

        # The Barthel fit stores the labeled shots concatenated as S then T.
        # Split them back into their two physical preparation arms for plotting.
        n_runs = int(ds_labeled["Ig"].sel(qubit=qname).shape[-1])
        y_all = np.asarray(fit.y_pca.values, dtype=float).ravel()
        y_s = y_all[:n_runs]
        y_t = y_all[n_runs : n_runs * 2]

        # These analytic densities are already on the pooled-mixture scale:
        #   density_S integrates to w_S
        #   density_T_no + density_T_dec integrates to w_T(no)+w_T(dec)
        # To compare like-for-like, the histogram bars must use the same pooled
        # area rather than normalizing S and T independently to unit area.
        xs = np.asarray(fit.density_grid.values, dtype=float)
        total = np.asarray(fit.density_total.values, dtype=float)
        S_comp = np.asarray(fit.density_S.values, dtype=float)
        T_no = np.asarray(fit.density_T_no.values, dtype=float)
        T_dec = np.asarray(fit.density_T_dec.values, dtype=float)
        w = np.asarray(fit.weights.values, dtype=float).ravel()
        w_S = float(w[0]) if w.size >= 1 else 1.0
        w_T = float(w[1] + w[2]) if w.size >= 3 else 1.0

        # Use a single shared binning so the S/T bars line up visually.
        valid_s = y_s[np.isfinite(y_s)]
        valid_t = y_t[np.isfinite(y_t)]
        pooled = np.concatenate([valid_s, valid_t]) if len(valid_s) + len(valid_t) > 0 else np.array([0.0, 1.0])
        bin_edges = np.histogram_bin_edges(pooled, bins=n_bins)

        s_centers, s_widths, s_density = _weighted_hist_density(valid_s, bin_edges, w_S)
        t_centers, t_widths, t_density = _weighted_hist_density(valid_t, bin_edges, w_T)

        # Histogram bars: pooled-density normalization to match the analytic curves.
        ax.bar(s_centers, s_density, width=s_widths, alpha=0.35, label="S shots", color="C0", align="center")
        ax.bar(t_centers, t_density, width=t_widths, alpha=0.35, label="T shots", color="C1", align="center")

        # Analytic Barthel model: keep the same component/total labels as before.
        ax.plot(xs, total, lw=2.0, color="black", label="Total (Barthel)")
        ax.plot(xs, S_comp, lw=1.2, ls="--", color="C0", label="S component")
        ax.plot(xs, T_no, lw=1.2, ls="--", color="C2", label="T (no decay)")
        ax.plot(xs, T_dec, lw=1.2, ls="--", color="C3", label="T (decay)")

        # The Barthel threshold lives in the normalized PCA coordinate used by
        # the analytic fit, so plot it directly on this axis.
        thr = float(np.asarray(fit.norm_ge_threshold.values).ravel()[0])
        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.3g}")

        # Keep the fitted mixture weights visible: they help interpret how much
        # of the triplet branch sits in the no-decay vs decay subcomponents.
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
) -> Figure:
    """Per-qubit labeled S/T histograms with fitted GMM Gaussian components.

    The histogram is plotted on the same physical rotated-readout axis used for
    ``I_threshold`` so the threshold line matches the state-update convention.
    """

    qnames = [q.name for q in qubits]
    n = len(qnames)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, qname in enumerate(qnames):
        ax = axes_flat[idx]
        fit = ds_gmm_fit.sel(qubit=qname)

        # Stored shot coordinates are the 1D PCA projections of the labeled S/T
        # clouds. We plot them after shifting onto the same physical readout axis
        # used for the final threshold/state update.
        y_g = np.asarray(fit.y_g.values, dtype=float).ravel()
        y_e = np.asarray(fit.y_e.values, dtype=float).ravel()
        y_g = y_g[np.isfinite(y_g)]
        y_e = y_e[np.isfinite(y_e)]

        offset = float(fit.readout_axis_offset) if "readout_axis_offset" in fit else 0.0
        y_g = y_g + offset
        y_e = y_e + offset

        m_S, s_S, w_S = (float(fit.gmm_mean_S) + offset, float(fit.gmm_std_S), float(fit.gmm_weight_S))
        m_T, s_T, w_T = (float(fit.gmm_mean_T) + offset, float(fit.gmm_std_T), float(fit.gmm_weight_T))
        thr = float(fit.I_threshold) if "I_threshold" in fit else float(fit.ge_threshold) + offset
        fid = float(fit.readout_fidelity) * 100.0

        # Plot in mV to match the rotated-I threshold shown elsewhere in the node.
        scale = 1e3
        y_g *= scale
        y_e *= scale
        m_S *= scale
        m_T *= scale
        s_S *= scale
        s_T *= scale
        thr *= scale

        # Use a shared x-range/binning so the weighted histograms and Gaussian
        # PDFs are directly comparable on a single pooled-density axis.
        lo = min(np.min(y_g), np.min(y_e), m_S - 4 * s_S, m_T - 4 * s_T, thr)
        hi = max(np.max(y_g), np.max(y_e), m_S + 4 * s_S, m_T + 4 * s_T, thr)
        xs = np.linspace(lo, hi, 600)
        bin_edges = np.linspace(lo, hi, n_bins + 1)

        s_centers, s_widths, s_density = _weighted_hist_density(y_g, bin_edges, w_S)
        t_centers, t_widths, t_density = _weighted_hist_density(y_e, bin_edges, w_T)

        # Histogram bars: pooled-density normalization to match the mixture PDFs.
        ax.bar(s_centers, s_density, width=s_widths, alpha=0.35, label="S shots", color="C0", align="center")
        ax.bar(t_centers, t_density, width=t_widths, alpha=0.35, label="T shots", color="C1", align="center")

        # GMM overlay: restore the original single-mixture view and labels.
        s_pdf = _scipy_norm.pdf(xs, m_S, s_S)
        t_pdf = _scipy_norm.pdf(xs, m_T, s_T)
        ax.plot(xs, w_S * s_pdf, lw=1.2, ls="--", color="C0", label="S component")
        ax.plot(xs, w_T * t_pdf, lw=1.2, ls="--", color="C1", label="T component")
        ax.plot(xs, w_S * s_pdf + w_T * t_pdf, lw=2.0, color="black", label="Total (GMM)")

        # The threshold is shown on the same physical mV axis as the histogram/PDFs.
        if np.isfinite(thr):
            ax.axvline(thr, color="r", ls="--", lw=1.5, label=f"Threshold = {thr:.3g} mV")

        ax.set_title(f"{qname}  (F = {fid:.1f} %)")
        ax.set_xlabel("Projected readout [mV]")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("PSB readout: GMM — labeled S/T histograms + Gaussian components")
    fig.tight_layout()
    return fig


def plot_all(
    ds_labeled: xr.Dataset,
    qubits: Sequence[Any],
    ds_fit: xr.Dataset,
    *,
    fit_results: Dict[str, Dict[str, Any]],
    analysis_model: str,
) -> Dict[str, Figure]:
    """Standard 06d plotting entry point returning all node figures."""

    # Keep the node-facing plotting API intentionally thin: the node hands over
    # processed data + fit results, and this helper returns the full figure dict.
    figures = {
        "iq_blobs": plot_labeled_iq_blobs(ds_labeled, qubits, fit_results),
    }

    if analysis_model == "barthel":
        figures["histogram"] = plot_labeled_histogram_barthel(ds_labeled, ds_fit, qubits)
    elif analysis_model == "gmm":
        figures["histogram"] = plot_labeled_histogram_gmm(ds_fit, qubits)
    else:
        raise ValueError(f"Unsupported analysis_model={analysis_model!r}.")

    return figures
