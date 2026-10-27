from typing import List

import numpy as np
import xarray as xr
from matplotlib import colormaps

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import ALPHA_RELIABLE_MAX, NOISE_FIT_F_REF_HZ


def plot_decay_curves(ds_fit: xr.Dataset, qubits: List[AnyTransmon]):
    """Normalized coherence vs total evolution time, one curve per number of pulses per window, with the fits."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    n_values = ds_fit.pulses_per_window.values
    colors = colormaps["viridis"](np.linspace(0, 0.9, len(n_values)))
    window_us = 1e-3 * float(ds_fit.window_ns)
    sequence = _sequence_name(ds_fit)
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        for i_n, n in enumerate(n_values):
            curve = fit.isel(pulses_per_window=i_n)
            t_us = 1e-3 * curve.time.values
            a, off = float(curve.amplitude), float(curve.offset)
            ax.plot(t_us, curve.coherence, "o", ms=3, color=colors[i_n], label=f"$N={n}$")
            ax.plot(t_us, (curve.fit - off) / a, "-", lw=1, color=colors[i_n])
        ax.set_title(qubit["qubit"])
        ax.set_xlabel(r"Total evolution time ($\mu$s)")
        ax.set_ylabel("Normalized coherence")
        top = ax.secondary_xaxis("top", functions=(lambda t: t / window_us, lambda m: m * window_us))
        top.set_xlabel("Windows", fontsize=8)
        ax.legend(title="Pulses", fontsize=10, title_fontsize=10, ncol=2)
        _add_floor_inset(ax, fit)
    grid.fig.suptitle(rf"{sequence} decay (window $T_w$ = {window_us:.2f} $\mu$s)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_t2_vs_pulses(ds_fit: xr.Dataset, qubits: List[AnyTransmon]):
    """T2 vs number of pulses per window, with the stretch exponent of each point and the 2T1 limit."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    T1 = {q.name: q.T1 for q in qubits}
    sequence = _sequence_name(ds_fit)
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        n_values = fit.pulses_per_window.values
        T2_us = 1e-3 * fit.T2.values
        ax.errorbar(n_values, T2_us, yerr=1e-3 * fit.T2_error, fmt="ko-", ms=4, capsize=2, label=r"$T_2$")
        _mark_floor_shifted(ax, n_values, T2_us, fit.floor_shifted.values)
        # Stretch exponent below each error bar
        T2_err_us = 1e-3 * fit.T2_error.values
        for n, T2, dT2, al in zip(n_values, T2_us, T2_err_us, fit.alpha.values):
            if np.isfinite(T2) and np.isfinite(al):
                ax.annotate(
                    rf"$\alpha$={al:.1f}",
                    (n, T2 - np.nan_to_num(dT2)),
                    textcoords="offset points",
                    xytext=(0, -4),
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8),
                )
        # No-DD baseline (06a_ramsey)
        T2_star = _T2_star_s(qubits, qubit["qubit"])
        if T2_star is not None:
            T2_star_us = 1e6 * T2_star
            ax.axhline(T2_star_us, color="tab:purple", ls=":", lw=1.2, label=r"$T_2^*$ (no DD)")
            ax.annotate(
                rf"$T_2^*$ = {T2_star_us:.1f} $\mu$s",
                (0.99, T2_star_us),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=8,
                color="tab:purple",
            )
        # T1 limit (05_T1)
        if T1.get(qubit["qubit"]):
            two_T1_us = 2e6 * T1[qubit["qubit"]]
            ax.axhline(two_T1_us, color="b", ls="-.", lw=1, label=r"$2T_1$ limit")
            ax.annotate(
                rf"$2T_1$ = {two_T1_us:.0f} $\mu$s",
                (0.01, two_T1_us),
                xycoords=("axes fraction", "data"),
                va="bottom",
                fontsize=8,
                color="b",
            )
        ax.set_title(qubit["qubit"])
        ax.set_xlabel(r"$\pi$ pulses per window $N$")
        ax.set_ylabel(rf"$T_2$ under {sequence} ($\mu$s)")
        _add_tau_axis(ax, n_values, fit.tau.values)
        ax.legend(fontsize=10)
    grid.fig.suptitle(rf"$T_2$ under {sequence} vs $\pi$ pulses per window")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_error_per_round(ds_fit: xr.Dataset, qubits: List[AnyTransmon]):
    """Average dephasing error per round over M rounds vs number of pulses per window (measured), with the error
    budget, the no-DD baseline and the optimal N."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    sequence = _sequence_name(ds_fit)
    window_us = 1e-3 * float(ds_fit.window_ns)
    num_rounds = int(ds_fit.num_rounds)
    budget = float(ds_fit.max_extra_error_per_round)
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        n_values = fit.pulses_per_window.values
        ax.errorbar(
            n_values,
            1e2 * fit.error_per_round,
            yerr=1e2 * fit.error_per_round_error,
            fmt="ko",
            ms=4,
            capsize=2,
            label=r"measured $\bar{p}_N$",
        )
        _mark_floor_shifted(ax, n_values, 1e2 * fit.error_per_round.values, fit.floor_shifted.values)
        # No-DD baseline from T2* (exponential decay assumed)
        T2_star = _T2_star_s(qubits, qubit["qubit"])
        if T2_star is not None:
            p_no_dd = 1e2 * (1 - np.exp(-1e-9 * float(ds_fit.window_ns) / T2_star)) / 2
            ax.axhline(p_no_dd, color="tab:purple", ls=":", lw=1.2, label=rf"no DD ($T_2^*$): {p_no_dd:.2f}%")
        if bool(fit.success):
            p_best = 1e2 * float(fit.error_per_round_best)
            budget_label = f"acceptable: within {1e2 * budget:g}% of best"
            ax.axhspan(p_best, p_best + 1e2 * budget, color="g", alpha=0.15, label=budget_label)
            n_sel = int(fit.pulses_per_window_selected)
            ax.axvline(n_sel, color="r", ls="--", lw=1, label=rf"optimal $N$ = {n_sel}")
        ax.set_title(qubit["qubit"])
        ax.set_xlabel(r"$\pi$ pulses per window $N$")
        ax.set_ylabel(r"Average dephasing error per round $\bar{p}_N$ (%)")
        _add_tau_axis(ax, n_values, fit.tau.values)
        ax.legend(fontsize=10)
    grid.fig.suptitle(
        rf"{sequence}: average dephasing error per round over $M$ = {num_rounds} rounds "
        rf"({window_us:.2f} $\mu$s windows) vs $\pi$ pulses per window"
    )
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_noise_spectrum(ds_fit: xr.Dataset, qubits: List[AnyTransmon]):
    """First-order dephasing noise spectrum S_f(f0) = 1 / (8 T_phi) vs DD filter frequency f0 = N / (2 window)."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    sequence = _sequence_name(ds_fit)
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        f0, psd, err, alpha = (fit[k].values for k in ("noise_frequency", "noise_psd", "noise_psd_error", "alpha"))
        reliable = alpha <= ALPHA_RELIABLE_MAX
        groups = [(reliable, "k", "data"), (~reliable, "none", rf"unreliable ($\alpha > {ALPHA_RELIABLE_MAX}$)")]
        for mask, face, label in groups:
            if np.any(mask & np.isfinite(psd)):
                ax.errorbar(
                    f0[mask],
                    psd[mask],
                    yerr=err[mask],
                    fmt="o",
                    ms=5,
                    mfc=face,
                    mec="k",
                    ecolor="k",
                    capsize=2,
                    label=label,
                )
        A, beta, C = (float(fit[f"noise_fit_{k}"]) for k in ("amplitude", "exponent", "floor"))
        if np.isfinite(A):
            f_line = np.geomspace(np.nanmin(f0), np.nanmax(f0), 200)
            ax.plot(f_line, A * (NOISE_FIT_F_REF_HZ / f_line) ** beta + C, "r--", lw=1, label="fit")
            dA, dbeta, dC = (float(fit[f"noise_fit_{k}_error"]) for k in ("amplitude", "exponent", "floor"))
            ax.text(
                0.98,
                0.98,
                rf"$A$ (1 MHz) = {A:.0f} $\pm$ {dA:.0f} Hz"
                "\n"
                rf"$\beta$ = {beta:.2f} $\pm$ {dbeta:.2f}"
                "\n"
                rf"$C$ = {C:.0f} $\pm$ {dC:.0f} Hz",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        elif np.count_nonzero(np.isfinite(psd)):
            ax.text(0.98, 0.98, "no reliable power-law fit", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not bool(fit.T1_subtracted):
            ax.text(0.02, 0.02, r"$T_1$ unknown: not subtracted", transform=ax.transAxes, fontsize=7, color="r")
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$S_f$ (Hz, one-sided)")
        ax.legend(fontsize=10, loc="lower left")
    grid.fig.suptitle(f"{sequence} dephasing noise spectrum")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _add_floor_inset(ax, fit: xr.Dataset):
    """Inset with the raw decay floor of each N; the main plot normalizes each curve to its own floor."""
    n_values, floor, err = fit.pulses_per_window.values, fit.offset.values, fit.offset_error.values
    shifted = fit.floor_shifted.values
    inset = ax.inset_axes([0.38, 0.68, 0.26, 0.29])
    inset.errorbar(n_values, floor, yerr=err, fmt="ko", ms=3, capsize=1.5, lw=0.8)
    if np.any(shifted):
        inset.plot(n_values[shifted], floor[shifted], "o", ms=5, mfc="white", mec="tab:orange", mew=1.2)
    inset.axhline(floor[0], color="gray", ls=":", lw=0.8)
    inset.set_xlabel(r"$N$", fontsize=7, labelpad=1)
    inset.set_ylabel("decay floor", fontsize=7, labelpad=1)
    inset.tick_params(labelsize=6, pad=1)


def _T2_star_s(qubits: List[AnyTransmon], name: str):
    """T2* of the qubit from the state (node 06a_ramsey) in s, or None if missing or invalid (<= 0)."""
    q = next((q for q in qubits if q.name == name), None)
    T2_star = getattr(q, "T2ramsey", None) if q is not None else None
    return float(T2_star) if T2_star is not None and np.isfinite(T2_star) and T2_star > 0 else None


def _mark_floor_shifted(ax, n_values, values, shifted):
    """Hollow markers on the points whose decay settles at a different floor than the fewest-pulse curve (excluded
    from the decision)."""
    if np.any(shifted):
        ax.plot(
            n_values[shifted],
            values[shifted],
            "o",
            ms=7,
            mfc="white",
            mec="tab:orange",
            mew=1.5,
            label=rf"excluded: decay floor $\neq$ $N$ = {n_values[0]} (leakage/heating?)",
        )


def _add_tau_axis(ax, n_values, tau_ns):
    """Pulse half-spacing tau on the top axis."""
    top = ax.secondary_xaxis("top")
    top.set_xticks(n_values)
    top.set_xticklabels([f"{t:.0f}" for t in tau_ns], fontsize=10, rotation=90)
    top.set_xlabel(r"$\tau$ (ns)", fontsize=12)


def _sequence_name(ds_fit: xr.Dataset) -> str:
    return str(ds_fit.sequence.values) if "sequence" in ds_fit.coords else "CPMG"
