"""Plotting for T1 ADE tracking (node T1_ADE)."""

from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibrate import QualibrationNode
from qualibration_libs.plotting import QubitGrid, grid_iter


def _sigma_clipped_from_ds(ds: xr.Dataset, qubits: list) -> tuple[dict, dict, dict]:
    sigma_T1_by_qubit = {q.name: ds.sigma_T1.sel(qubit=q.name).values for q in qubits}
    sigma_T1_boot_by_qubit = {q.name: ds.sigma_T1_boot.sel(qubit=q.name).values for q in qubits}
    clipped_by_qubit = {q.name: ds.clipped.sel(qubit=q.name).values for q in qubits}
    return sigma_T1_by_qubit, sigma_T1_boot_by_qubit, clipped_by_qubit


def _sigma_clipped_from_fits(fits: Dict) -> tuple[dict, dict, dict]:
    sigma_T1_by_qubit = {k: np.asarray(v["sigma_T1_us"]) for k, v in fits.items()}
    sigma_T1_boot_by_qubit = {k: np.asarray(v["sigma_T1_boot_us"]) for k, v in fits.items()}
    clipped_by_qubit = {k: np.asarray(v["clipped"]) for k, v in fits.items()}
    return sigma_T1_by_qubit, sigma_T1_boot_by_qubit, clipped_by_qubit


def plot_raw_data_with_fit(
    node: QualibrationNode,
    grid_size: int = 8,
    bin_width_s: float = 0.25,
) -> Dict[str, Figure]:
    """Plot T1 vs lab time (analytical and bootstrap) and ADE wait-time fit per qubit.

    Reads ``ds_fit``, ``fit_results``, qubits, and ``t0_ns`` from the node.

    Returns
    -------
    dict[str, Figure]
        ``T1_vs_lab_time``, ``T1_vs_lab_time_bootstrap``, and ``wait_time_image``.
    """
    ds = node.results["ds_fit"]
    qubits = node.namespace["qubits"]
    fits: Optional[Dict] = node.results.get("fit_results")
    t0_ns = node.parameters.t0_ns

    if fits is not None:
        sigma_T1_by_qubit, sigma_T1_boot_by_qubit, clipped_by_qubit = _sigma_clipped_from_fits(fits)
    else:
        sigma_T1_by_qubit, sigma_T1_boot_by_qubit, clipped_by_qubit = _sigma_clipped_from_ds(ds, qubits)

    mean_dt_ms = float(ds.attrs.get("mean_dt_s", np.nan)) * 1e3
    figures: Dict[str, Figure] = {}

    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=grid_size)
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        plot_individual_T1_lab_time(
            ax,
            qubit_name,
            ds.time_stamp.sel(qubit=qubit_name).values,
            ds.estimated_T1.sel(qubit=qubit_name).values,
            sigma_T1_by_qubit[qubit_name],
            ds.estimated_gamma.sel(qubit=qubit_name).values,
            clipped_by_qubit[qubit_name],
            bin_width_s=bin_width_s,
            band_alpha=0.5,
            line_alpha=0.75,
        )
    grid.fig.suptitle("T1 vs laboratory time \n (FPGA analytical $\sigma$)")
    grid.fig.tight_layout()
    figures["T1_vs_lab_time"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=grid_size)
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        plot_individual_T1_lab_time(
            ax,
            qubit_name,
            ds.time_stamp.sel(qubit=qubit_name).values,
            ds.estimated_T1.sel(qubit=qubit_name).values,
            sigma_T1_boot_by_qubit[qubit_name],
            ds.estimated_gamma.sel(qubit=qubit_name).values,
            clipped_by_qubit[qubit_name],
            bin_width_s=bin_width_s,
            line_color="tab:blue",
            band_color="tab:blue",
            band_alpha=0.5,
            line_alpha=0.75,
            sigma_label=r"$T_1 \pm \sigma_\mathrm{boot}$",
        )
    grid.fig.suptitle("T1 vs laboratory time \n (bootstrap $\sigma$)")
    grid.fig.tight_layout()
    figures["T1_vs_lab_time_bootstrap"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=grid_size)
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        plot_individual_ade_wait_with_fit(
            ax,
            qubit_name,
            ds.dt_used.sel(qubit=qubit_name).values,
            ds.estimated_gamma.sel(qubit=qubit_name).values,
            ds.P0.sel(qubit=qubit_name).values,
            ds.P1.sel(qubit=qubit_name).values,
            ds.P3.sel(qubit=qubit_name).values,
            sigma_T1_by_qubit[qubit_name],
            clipped_by_qubit[qubit_name],
            t0_ns,
        )
    title = r"$P_{|1\rangle}$ vs wait time (best $\sigma$ repetition)"
    if np.isfinite(mean_dt_ms):
        title += f"\n({mean_dt_ms:.1f} ms/rep)"
    grid.fig.suptitle(title)
    grid.fig.tight_layout(rect=[0, 0, 1, 0.92])
    figures["wait_time_image"] = grid.fig

    return figures


def plot_individual_T1_lab_time(
    ax: Axes,
    qubit_name: str,
    t_s,
    T1_us,
    sigma_T1_us,
    gamma_us,
    clipped,
    bin_width_s: float = 0.25,
    line_color: str = "#b22222",
    band_color: str = "tab:red",
    band_alpha: float = 0.08,
    line_alpha: float = 0.75,
    sigma_label: str = r"on-FPGA $T_1 \pm \sigma$",
) -> None:
    """Paper-style panel: T1 line, ±σ band, and binned median T1."""
    valid = ~np.asarray(clipped) & np.isfinite(T1_us) & np.isfinite(sigma_T1_us)
    gamma_all = np.asarray(gamma_us)
    t_v = np.asarray(t_s)[valid]
    T1_v = np.asarray(T1_us)[valid]
    sig_v = np.asarray(sigma_T1_us)[valid]
    order = np.argsort(t_v)
    t_v, T1_v, sig_v = t_v[order], T1_v[order], sig_v[order]

    t_all = np.asarray(t_s)
    t_max = np.nanmax(t_all) if np.any(np.isfinite(t_all)) else bin_width_s
    bin_edges = np.arange(0, t_max + bin_width_s, bin_width_s)
    bin_idx = np.digitize(t_all, bin_edges)
    bin_centers, bin_T1 = [], []
    for b in range(1, len(bin_edges)):
        mask = (bin_idx == b) & valid
        if mask.sum() == 0:
            continue
        bin_centers.append(0.5 * (bin_edges[b - 1] + bin_edges[b]))
        bin_T1.append(1.0 / np.nanmedian(gamma_all[mask]))

    if t_v.size:
        ax.fill_between(
            t_v, T1_v - sig_v, T1_v + sig_v, color=band_color, alpha=band_alpha, linewidth=0, zorder=1
        )
        ax.plot(
            t_v, T1_v, color=line_color, linewidth=0.6, alpha=line_alpha,
            label=sigma_label, zorder=2,
        )

    if bin_centers:
        ax.plot(
            bin_centers,
            bin_T1,
            color="0.45",
            linewidth=1.0,
            zorder=3,
            label=rf"$\langle T_1 \rangle$ in {int(bin_width_s * 1e3)} ms",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$T_1$ ($\mu$s)")
    ax.set_title(qubit_name)
    ax.grid(False)
    ax.legend(fontsize=7, loc="upper right")


def _add_ade_fit_text(ax: Axes, T1_us: float, sigma_T1_us: float) -> None:
    ax.text(
        0.1,
        0.9,
        f"T1 = {T1_us:.1f} ± {sigma_T1_us:.1f} µs",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )


def plot_individual_ade_wait_with_fit(
    ax: Axes,
    qubit_name: str,
    dt_us: np.ndarray,
    gamma_us: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
    P3: np.ndarray,
    sigma_T1_us: np.ndarray,
    clipped: np.ndarray,
    t0_ns: int,
) -> None:
    """Best low-σ repetition: 3 ADE points and exponential curve through them."""
    t0_us = t0_ns * 1e-3
    sigma_for_selection = np.where(clipped, np.inf, sigma_T1_us)
    best_idx = int(np.nanargmin(sigma_for_selection))
    dt_best = dt_us[best_idx]
    gamma_best = gamma_us[best_idx]
    T1_best = 1.0 / gamma_best
    sigma_best = sigma_T1_us[best_idx]

    t_pts = np.array([t0_us, t0_us + dt_best, t0_us + 3 * dt_best])
    P_pts = np.array([P0[best_idx], P1[best_idx], P3[best_idx]])

    e0 = np.exp(-gamma_best * t_pts[0])
    e3 = np.exp(-gamma_best * t_pts[2])
    A_fit = (P_pts[0] - P_pts[2]) / (e0 - e3)
    C_fit = P_pts[0] - A_fit * e0

    t_smooth = np.linspace(0, t_pts[-1] * 1.15, 300)
    P_smooth = A_fit * np.exp(-gamma_best * t_smooth) + C_fit

    ax.plot(t_pts, P_pts, "o", color="tab:blue", markersize=6, label="3 ADE points")
    ax.plot(t_smooth, P_smooth, "r--", linewidth=1, label="ADE fit")
    ax.set_xlabel("Wait time (µs)")
    ax.set_ylabel(r"$P_{|1\rangle}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(qubit_name)
    ax.legend(fontsize=7)
    ax.grid(False)
    _add_ade_fit_text(ax, T1_best, sigma_best)
