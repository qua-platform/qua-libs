"""Plotting utilities for geometric CZ error amplification."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - target)))


def _reference_amplitude(
    amplitudes: np.ndarray,
    exchange_amplitude_center: float | None,
) -> float:
    if exchange_amplitude_center is not None and np.isfinite(exchange_amplitude_center):
        return float(exchange_amplitude_center)
    return float(np.mean(amplitudes))


def _branch_phases_vs_n(
    data: xr.DataArray,
    amp_plot: float,
    *,
    signal_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ramsey phases φ0, φ1 from parity I/Q at fixed exchange amplitude."""
    phi_out: list[np.ndarray] = []
    for ctrl_state in (0, 1):
        signal_i = data.sel(
            control_state=ctrl_state,
            analysis_axis=0,
            exchange_amplitude=amp_plot,
        ).values
        signal_q = data.sel(
            control_state=ctrl_state,
            analysis_axis=1,
            exchange_amplitude=amp_plot,
        ).values
        phi = np.arctan2(
            np.asarray(signal_q, dtype=float) - signal_center,
            np.asarray(signal_i, dtype=float) - signal_center,
        )
        phi_out.append(phi)
    return phi_out[0], phi_out[1]


def _plot_left_center_slice(
    ax: plt.Axes,
    data: xr.DataArray,
    ds_fit: xr.Dataset | None,
    name: str,
    amp_plot: float,
    num_gates: np.ndarray,
    *,
    signal_center: float,
) -> None:
    """Explain the center panel: φ0, φ1 → Δφ slice; twin axis cos(Δφ) used for chevron / right panel.

    Δφ is the same array stored in ``chevron_signal_*`` (wrapped φ1−φ0). ``cos(Δφ)`` is averaged over
    ``N`` at each voltage to build ⟨cos(Δφ)⟩ on the right; the green DE fit runs on the full 2D cos grid.
    """
    phi0, phi1 = _branch_phases_vs_n(data, amp_plot, signal_center=signal_center)
    dphi = np.arctan2(np.sin(phi1 - phi0), np.cos(phi1 - phi0))

    key = f"chevron_signal_{name}"
    if ds_fit is not None and key in ds_fit.data_vars:
        dphi_ds = (
            ds_fit[key]
            .sel(exchange_amplitude=amp_plot, method="nearest")
            .values.astype(np.float64)
            .ravel()
        )
        if len(dphi_ds) == len(num_gates):
            dphi = dphi_ds

    ax.plot(
        num_gates,
        phi0,
        linestyle="none",
        marker="o",
        ms=3,
        color="C0",
        alpha=0.65,
        label="φ0 = atan2(Q−c, I−c)  |0⟩",
    )
    ax.plot(
        num_gates,
        phi1,
        linestyle="none",
        marker="s",
        ms=3,
        color="C1",
        alpha=0.65,
        label="φ1  |1⟩",
    )
    ax.plot(
        num_gates,
        dphi,
        "-",
        color="k",
        lw=1.8,
        alpha=0.85,
        label="Δφ (wrap(φ1−φ0))",
    )
    ax.axhline(np.pi, color="0.45", ls=":", lw=0.75, alpha=0.7)
    ax.axhline(-np.pi, color="0.45", ls=":", lw=0.75, alpha=0.7)
    ax.set_ylabel("phase (rad)", color="k")
    ax.tick_params(axis="y", labelcolor="k")
    ax.set_ylim(-np.pi * 1.12, np.pi * 1.12)

    ax_cos = ax.twinx()
    cos_dphi = np.cos(dphi)
    ax_cos.plot(
        num_gates,
        cos_dphi,
        "--",
        color="C2",
        lw=1.2,
        alpha=0.9,
        label="cos(Δφ)",
    )
    ax_cos.axhline(-1.0, color="C2", ls=":", lw=0.8, alpha=0.45)
    ax_cos.set_ylabel("cos(Δφ)", color="C2")
    ax_cos.tick_params(axis="y", labelcolor="C2")
    ax_cos.set_ylim(-1.12, 1.12)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_cos.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=6.5, ncol=2)


def _plot_chevron_2d_map(
    fig: plt.Figure,
    ax: plt.Axes,
    ds_fit: xr.Dataset | None,
    name: str,
    *,
    reference_amplitude: float,
    slice_amplitude: float,
    fit_result: dict[str, Any],
    num_gates: np.ndarray,
    amplitudes: np.ndarray,
) -> None:
    """Plot wrapped Δφ(V, N) (same quantity averaged for ⟨Δφ⟩ on the right)."""
    key = f"chevron_signal_{name}"
    if ds_fit is None or key not in ds_fit.data_vars:
        ax.set_title(f"{name} - no chevron map")
        ax.set_xlim(float(num_gates.min()), float(num_gates.max()))
        ax.set_ylim(float(amplitudes.min()), float(amplitudes.max()))
        return

    signal_2d = ds_fit[key].values.astype(np.float64)
    im = ax.pcolormesh(
        num_gates,
        amplitudes,
        signal_2d,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Δφ (rad)")

    ax.axhline(
        reference_amplitude,
        color="0.2",
        ls="--",
        lw=1.0,
    )
    ax.axhline(
        slice_amplitude,
        color="cyan",
        ls="-.",
        lw=1.1,
        alpha=0.85,
        label="left-panel slice A",
    )
    if fit_result.get("success"):
        ax.axhline(
            fit_result["optimal_amplitude"],
            color="red",
            ls="-",
            lw=1.2,
        )

    ax.legend(loc="upper left", fontsize=5, framealpha=0.88)

    ax.set_title(f"{name} - Δφ chevron")
    ax.set_xlabel("Number of raw CPhase gates")
    ax.set_ylabel("Exchange amplitude (V)")
    ax.set_xlim(float(num_gates.min()), float(num_gates.max()))
    ax.set_ylim(float(amplitudes.min()), float(amplitudes.max()))


def _plot_raw_parity(
    ax: plt.Axes,
    data: xr.DataArray,
    amp_plot: float,
    num_gates: np.ndarray,
    *,
    signal_center: float,
) -> None:
    """Plot raw parity I and Q signals vs N at fixed exchange amplitude.

    Shows the conditional expectation values directly (before any arctan2
    transformation) for both control states and both analysis quadratures.
    """
    colors = {0: "C0", 1: "C1"}
    labels = {0: "|0⟩", 1: "|1⟩"}

    for ctrl in (0, 1):
        i_vals = data.sel(
            control_state=ctrl, analysis_axis=0, exchange_amplitude=amp_plot
        ).values.astype(float)
        q_vals = data.sel(
            control_state=ctrl, analysis_axis=1, exchange_amplitude=amp_plot
        ).values.astype(float)
        ax.plot(
            num_gates, i_vals,
            "-o", ms=3, color=colors[ctrl], lw=1.2,
            label=f"I ctrl {labels[ctrl]}",
        )
        ax.plot(
            num_gates, q_vals,
            "--s", ms=3, color=colors[ctrl], lw=1.2, alpha=0.7,
            label=f"Q ctrl {labels[ctrl]}",
        )

    ax.axhline(signal_center, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="0.7", ls=":", lw=0.6, alpha=0.4)
    ax.axhline(1.0, color="0.7", ls=":", lw=0.6, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Number of raw CPhase gates")
    ax.set_ylabel("Parity signal")
    ax.legend(loc="best", fontsize=6, ncol=2)


def plot_raw_state_panels(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """2D maps of average state for each (control_state, analysis_axis) experiment.

    Produces a figure with *n_pairs* rows and 4 columns — one column per
    (control_state, analysis_axis) combination — showing pcolormesh of the
    raw parity signal as a function of exchange amplitude (y) and gate count (x).
    """
    n_pairs = len(qubit_pairs)
    experiments = [
        (0, 0, "|0⟩ ctrl · I quad (ax=0)"),
        (0, 1, "|0⟩ ctrl · Q quad (ax=1)"),
        (1, 0, "|1⟩ ctrl · I quad (ax=0)"),
        (1, 1, "|1⟩ ctrl · Q quad (ax=1)"),
    ]

    fig, axes = plt.subplots(
        n_pairs,
        4,
        figsize=(20, max(4.0, 3.8 * n_pairs)),
        squeeze=False,
    )

    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(float)
    num_gates = ds_raw.coords["num_cphase_gates"].values.astype(float)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"

        for col, (ctrl, ax_idx, label) in enumerate(experiments):
            ax = axes[pair_idx, col]

            if var_name not in ds_raw.data_vars:
                ax.set_title(f"{name} - no data")
                continue

            panel = (
                ds_raw[var_name]
                .sel(control_state=ctrl, analysis_axis=ax_idx)
                .values.astype(float)
            )  # shape: (n_amplitudes, n_gates)

            im = ax.pcolormesh(
                num_gates,
                amplitudes,
                panel,
                cmap="RdBu_r",
                shading="auto",
            )
            fig.colorbar(im, ax=ax, label="avg state")

            ax.set_title(f"{name}\n{label}", fontsize=8)
            ax.set_xlabel("Number of raw CPhase gates")
            ax.set_ylabel("Exchange amplitude (V)")

    fig.suptitle(
        "Geometric CZ Error Amplification — Raw State Maps", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
    exchange_amplitude_center: float | None = None,
) -> plt.Figure:
    """Create the error-amplification summary figure."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs,
        4,
        figsize=(24, max(4.5, 4.2 * n_pairs)),
        squeeze=False,
    )

    amplitudes = ds_raw.coords["exchange_amplitude"].values
    num_gates = ds_raw.coords["num_cphase_gates"].values
    ref_amp = _reference_amplitude(amplitudes, exchange_amplitude_center)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"
        ax_parity = axes[pair_idx, 0]
        ax_raw = axes[pair_idx, 1]
        ax_map = axes[pair_idx, 2]
        ax_fit = axes[pair_idx, 3]
        if var_name not in ds_raw.data_vars:
            for ax in (ax_parity, ax_raw, ax_map, ax_fit):
                ax.set_title(f"{name} - no data")
            continue

        data = ds_raw[var_name]
        r = fit_results.get(name, {})
        amp_plot = amplitudes[_nearest_index(amplitudes, ref_amp)]

        _plot_raw_parity(
            ax_parity,
            data,
            amp_plot,
            num_gates,
            signal_center=quadrature_signal_center,
        )
        ax_parity.set_title(f"{name} — raw parity at A={amp_plot:.4f} V")

        _plot_left_center_slice(
            ax_raw,
            data,
            ds_fit,
            name,
            amp_plot,
            num_gates,
            signal_center=quadrature_signal_center,
        )
        ax_raw.set_title(
            f"{name} — slice at A={amp_plot:.4f} V (same row as dashed line in Δφ map)"
        )
        ax_raw.set_xlabel("Number of raw CPhase gates")

        _plot_chevron_2d_map(
            fig,
            ax_map,
            ds_fit,
            name,
            reference_amplitude=ref_amp,
            slice_amplitude=amp_plot,
            fit_result=r,
            num_gates=num_gates,
            amplitudes=amplitudes,
        )

        ax_cos = ax_fit.twinx()
        if ds_fit is not None:
            mean_signal_fit_key = f"mean_signal_fit_{name}"
            mean_signal_key = f"mean_signal_{name}"

            if mean_signal_key in ds_fit.data_vars:
                mean_cos = ds_fit[mean_signal_key].values
                ax_cos.plot(
                    amplitudes,
                    mean_cos,
                    "o",
                    color="C2",
                    ms=4,
                    alpha=0.5,
                    label="⟨cos(Δφ)⟩_N",
                )
            if mean_signal_fit_key in ds_fit.data_vars:
                fit_vals = ds_fit[mean_signal_fit_key].values
                if len(fit_vals) == len(amplitudes):
                    ax_cos.plot(
                        amplitudes,
                        fit_vals,
                        "-",
                        color="C2",
                        lw=1.2,
                        label="Chevron 1D Fit",
                        zorder=5,
                    )
            ax_cos.set_ylabel("⟨cos(Δφ)⟩_N", color="C2")
            ax_cos.tick_params(axis="y", labelcolor="C2")
            ax_cos.set_ylim(-1.1, 1.1)
            ax_cos.grid(True, alpha=0.2)

        ax_fit.set_yticks([])
        ax_fit.set_ylabel("")

        h2, l2 = ax_cos.get_legend_handles_labels()
        ax_fit.legend(h2, l2, loc="best", fontsize=7)

        if r.get("success") and ds_fit is not None:
            ax_fit.axvline(
                ref_amp,
                color="0.45",
                ls="--",
                lw=1.0,
                label="previous A",
            )
            ax_fit.axvline(
                r["optimal_amplitude"],
                color="red",
                ls="-",
                lw=1.2,
                label=f"A*={r['optimal_amplitude']:.4f} V",
            )
            h1, l1 = ax_fit.get_legend_handles_labels()
            h2, l2 = ax_cos.get_legend_handles_labels()
            ax_fit.legend(h1 + h2, l1 + l2, loc="best", fontsize=7, ncol=2)

        ax_fit.set_title(f"{name} - mean cos(Δφ) vs amplitude")
        ax_fit.set_xlabel("Exchange amplitude (V)")

    fig.suptitle("Geometric CZ Error Amplification", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
