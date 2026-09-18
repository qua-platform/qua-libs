"""Plotting functions for XEB experiments.

Generates publication-quality figures for XEB results:
    - Fidelity vs depth (scatter + exponential fit)
    - Measured state probability heatmaps
    - Purity decay
    - 2Q unitary estimation diagnostics
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .analysis import _exponential_decay


def plot_xeb_fidelity(
    depths: np.ndarray,
    linear_fid: np.ndarray,
    log_fid: np.ndarray,
    fit_linear: dict,
    fit_log: dict,
    label: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot linear and log XEB fidelity vs circuit depth.

    Parameters
    ----------
    depths : ndarray
        Circuit depths.
    linear_fid : ndarray, shape (n_depths,)
        Mean linear XEB fidelity per depth.
    log_fid : ndarray, shape (n_depths,)
        Mean log XEB fidelity per depth.
    fit_linear : dict
        Fit results {"a", "r", "a_err", "r_err"} for linear fidelity.
    fit_log : dict
        Fit results for log fidelity.
    label : str
        Identifier for the qubit / qubit pair.
    ax : Axes or None
        If provided, plot on this axes. Otherwise create a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    d_fit = np.linspace(depths[0], depths[-1], 200)

    ax.scatter(depths, linear_fid, marker="o", s=30, label="Linear XEB", alpha=0.7)
    ax.scatter(depths, log_fid, marker="s", s=30, label="Log XEB", alpha=0.7)

    if not np.isnan(fit_linear["r"]):
        ax.plot(
            d_fit,
            _exponential_decay(d_fit, fit_linear["a"], fit_linear["r"]),
            "--",
            label=f"Linear fit: r={fit_linear['r']:.4f}±{fit_linear['r_err']:.4f}",
        )
    if not np.isnan(fit_log["r"]):
        ax.plot(
            d_fit,
            _exponential_decay(d_fit, fit_log["a"], fit_log["r"]),
            "--",
            label=f"Log fit: r={fit_log['r']:.4f}±{fit_log['r_err']:.4f}",
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("XEB Fidelity")
    ax.set_title(f"XEB Fidelity — {label}" if label else "XEB Fidelity")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_state_heatmap(
    depths: np.ndarray,
    measured_probs: np.ndarray,
    label: str = "",
    state_labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot measured state probabilities as a heatmap.

    Parameters
    ----------
    depths : ndarray
        Circuit depths.
    measured_probs : ndarray, shape (n_sequences, n_depths, dim)
        Measured probabilities.
    label : str
        Identifier.
    state_labels : list of str or None
        Labels for each state dimension. Defaults to ["|0⟩", "|1⟩"] or
        ["|00⟩", "|01⟩", "|10⟩", "|11⟩"].

    Returns
    -------
    Figure
    """
    dim = measured_probs.shape[-1]
    if state_labels is None:
        if dim == 2:
            state_labels = ["|0⟩", "|1⟩"]
        else:
            state_labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    n_states = len(state_labels)
    fig, axes = plt.subplots(1, n_states, figsize=(4 * n_states, 5), squeeze=False)

    for i, (ax, sl) in enumerate(zip(axes[0], state_labels)):
        data = measured_probs[:, :, i]
        im = ax.pcolormesh(
            depths,
            np.arange(data.shape[0]),
            data,
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax.set_xlabel("Depth")
        ax.set_ylabel("Sequence")
        ax.set_title(f"P({sl})")
        fig.colorbar(im, ax=ax)

    suptitle = f"State Probabilities — {label}" if label else "State Probabilities"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def plot_purity(
    depths: np.ndarray,
    purity: np.ndarray,
    fit_purity: dict,
    label: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot sqrt(purity) vs depth with exponential fit.

    Parameters
    ----------
    depths : ndarray
        Circuit depths.
    purity : ndarray, shape (n_depths,)
        Mean speckle purity per depth.
    fit_purity : dict
        Fit results {"a", "r", "a_err", "r_err"}.
    label : str
        Identifier.
    ax : Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    sqrt_pur = np.sqrt(np.clip(purity, 0, None))
    ax.scatter(depths, sqrt_pur, marker="^", s=30, alpha=0.7, label="√Purity")

    if not np.isnan(fit_purity["r"]):
        d_fit = np.linspace(depths[0], depths[-1], 200)
        ax.plot(
            d_fit,
            _exponential_decay(d_fit, fit_purity["a"], fit_purity["r"]),
            "--",
            label=f"Fit: r_pu={fit_purity['r']:.4f}±{fit_purity['r_err']:.4f}",
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("√(Speckle Purity)")
    ax.set_title(f"Purity Decay — {label}" if label else "Purity Decay")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_unitary_estimation(
    estimation_result: dict,
    label: str = "",
) -> plt.Figure:
    """Plot 2Q unitary estimation results.

    Shows a bar chart of the estimated fSim parameters.

    Parameters
    ----------
    estimation_result : dict
        Output from `estimate_2q_unitary()`.
    label : str
        Identifier.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    param_names = ["θ_iswap", "φ_cphase", "φ_rz1", "φ_rz2"]
    param_keys = ["theta_iswap", "phi_cphase", "phi_rz1", "phi_rz2"]
    values = [estimation_result[k] for k in param_keys]

    bars = ax.bar(param_names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel("Angle (rad)")
    ax.set_title(
        f"2Q Unitary Estimation — {label}\n"
        f"XEB Fidelity = {estimation_result['fidelity']:.4f}"
        if label
        else f"2Q Unitary Estimation\n"
        f"XEB Fidelity = {estimation_result['fidelity']:.4f}"
    )
    ax.axhline(y=np.pi, color="gray", linestyle=":", alpha=0.5, label="π")
    ax.axhline(y=np.pi / 2, color="gray", linestyle="--", alpha=0.5, label="π/2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig
