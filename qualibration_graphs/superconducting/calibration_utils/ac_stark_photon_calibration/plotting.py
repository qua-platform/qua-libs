from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import fit_fringe

__all__ = [
    "plot_fringes",
    "plot_phase_vs_time",
    "plot_photon_calibration",
    "plot_dephasing_vs_photon_number",
]


def _amplitude_colours(scales: np.ndarray) -> List:
    span = float(np.ptp(scales))
    lowest = float(np.min(scales))
    normalised = (scales - lowest) / span if span > 0 else np.zeros_like(scales)
    return [plt.get_cmap("viridis")(value) for value in normalised]


def plot_fringes(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Map the fringes at the longest free evolution time: tone amplitude against the phase of the second
    pi/2, coloured by the signal. Each row is one fringe; the Stark shift moves its peak sideways (as
    amplitude squared) and dephasing fades it. Crosses mark the peak of the cosine fitted to each row."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    signal_name = "state" if "state" in ds else "I"
    longest = int(np.argmax(ds.tau.values))
    tau_ns = float(ds.tau.values[longest])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        ds_q = ds.sel(qubit=name).isel(tau=longest)
        fit_q = fits.sel(qubit=name)
        phases = np.asarray(ds_q.phase.values, dtype=float)
        scales = np.asarray(ds_q.amp_scale.values, dtype=float)
        signal = np.asarray(ds_q[signal_name].transpose("amp_scale", "phase").values, dtype=float)
        mesh = ax.pcolormesh(phases, scales, signal, shading="nearest", cmap="RdBu_r")
        grid.fig.colorbar(mesh, ax=ax, pad=0.1, label="State population" if signal_name == "state" else "I [V]")

        usable = np.asarray(fit_q["fringe_usable"].isel(tau=longest).values, dtype=float) > 0
        n_bar = np.asarray(fit_q["n_bar"].values, dtype=float)
        for index, scale in enumerate(scales):
            _, fringe_phase, _ = fit_fringe(phases, signal[index])
            peak = (-fringe_phase / (2 * np.pi)) % 1.0
            ax.plot(peak, scale, marker="x", ms=9, mew=2, color="k" if usable[index] else "gray")
        # n̄ at the tone for each row, on the right-hand axis; rows dropped from the fit are marked.
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(scales)
        right.set_yticklabels(
            [(f"{n:.3f}" if np.isfinite(n) else "–") + ("" if ok else " ✗") for n, ok in zip(n_bar, usable)]
        )
        right.set_ylabel("n̄ at the tone (✗ = dropped)")

        ax.set_xlabel("Phase of the second π/2 [turns]")
        ax.set_ylabel("Tone amplitude scale")
        ax.set_title(f"{name} — τ = {tau_ns:.0f} ns")

    grid.fig.suptitle(f"AC Stark photon calibration (fringes at τ = {tau_ns:.0f} ns; × = fitted fringe peak)")
    grid.fig.set_size_inches(15, 9)
    _enlarge_fonts(grid.fig)
    grid.fig.tight_layout()
    return grid.fig


def plot_phase_vs_time(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the accumulated fringe phase against the free evolution time, with the fitted lines.

    This is the figure that makes the photon number falsifiable. A real Stark phase grows in
    proportion to the time the tone was on, so every line here has to be straight and every line has
    to be steeper than the one below it. A set of flat lines means the tone is not doing what the
    sequence assumes, whatever the fit quality against power says.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        fit_q = fits.sel(qubit=name)
        taus = np.asarray(ds.tau.values, dtype=float)
        scales = np.asarray(ds.amp_scale.values, dtype=float)
        colours = _amplitude_colours(scales)
        # The analysis has already put the phase, the slope and the intercept into one sign
        # convention, so the figure applies no sign of its own.
        for index, scale in enumerate(scales):
            if scale == 0:
                continue
            phase = np.asarray(fit_q["fringe_phase_shift"].isel(amp_scale=index).values, dtype=float)
            ax.plot(taus, phase, ls="", marker="o", ms=3.5, color=colours[index])
            slope = float(fit_q["delta_omega"].values[index])
            intercept = float(fit_q["phase_intercept"].values[index])
            if np.isfinite(slope) and np.isfinite(intercept):
                line = np.linspace(0, taus.max() * 1.05, 50)
                ax.plot(line, slope * line * 1e-9 + intercept, ls="-", lw=1, color=colours[index])
        min_tau_ns = float(fit_q["min_fitted_tau_ns"].values)
        if np.isfinite(min_tau_ns) and min_tau_ns > taus.min():
            ax.axvspan(0, min_tau_ns, color="gray", alpha=0.12, lw=0)
            ax.text(
                min_tau_ns, ax.get_ylim()[1], "transient, not fitted ", fontsize=10, va="top", ha="right", color="gray"
            )
        colour_scale = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(scales.min(), scales.max()))
        grid.fig.colorbar(colour_scale, ax=ax, label="Tone amplitude scale")
        ax.set_xlabel("Free evolution time τ [ns]")
        ax.set_ylabel("Accumulated fringe phase [rad]")
        linearity = float(fit_q["phase_linearity_r_squared"].values)
        ax.set_title(f"{name} — phase linearity R² = {linearity:.3f}")
        ax.axhline(0, color="gray", lw=0.6, ls=":")

    grid.fig.suptitle("AC Stark photon calibration (phase against free evolution time; slope is the Stark shift)")
    grid.fig.set_size_inches(15, 9)
    _enlarge_fonts(grid.fig)
    grid.fig.tight_layout()
    return grid.fig


def plot_photon_calibration(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot n̄ at the tone frequency against tone power, by both routes, with the fitted line, and the
    same line carried over to the readout frequency and extended to the operating power."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        fit_q = fits.sel(qubit=name)
        power_dbm = np.asarray(fit_q["tone_power_dbm"].values, dtype=float)
        n_bar = np.asarray(fit_q["n_bar"].values, dtype=float)
        n_gamma = np.asarray(fit_q["n_bar_from_gamma"].values, dtype=float)
        finite = np.isfinite(power_dbm) & np.isfinite(n_bar)
        ax.plot(power_dbm[finite], n_bar[finite], ls="", marker="o", ms=4, color="C0", label="from the Stark phase")
        finite_g = np.isfinite(power_dbm) & np.isfinite(n_gamma)
        ax.plot(
            power_dbm[finite_g],
            n_gamma[finite_g],
            ls="",
            marker="s",
            mfc="none",
            ms=4,
            color="C2",
            label="from the dephasing",
        )

        at_tone = float(fit_q["photons_per_mw_at_tone"].values)
        at_readout = float(fit_q["photons_per_mw"].values)
        if np.isfinite(at_tone) and finite.sum() >= 2:
            line_dbm = np.linspace(power_dbm[finite].min(), float(fit_q["operating_power_dbm"].values), 200)
            ax.plot(line_dbm, at_tone * 10 ** (line_dbm / 10), ls="-", lw=1.5, color="C0", label="fit, at the tone")
            ax.plot(
                line_dbm,
                at_readout * 10 ** (line_dbm / 10),
                ls="-",
                lw=1.5,
                color="C3",
                label="at the readout frequency",
            )

        operating_dbm = float(fit_q["operating_power_dbm"].values)
        if np.isfinite(operating_dbm):
            ax.axvline(operating_dbm, color="C3", lw=0.9, ls="--")
            ax.text(operating_dbm, ax.get_ylim()[1], " operating", fontsize=10, color="C3", va="top", rotation=90)
        single_photon_dbm = float(fit_q["single_photon_power_dbm"].values)
        if np.isfinite(single_photon_dbm):
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")

        ax.set_yscale("log")
        ax.set_xlabel("Stark tone power [dBm]")
        ax.set_ylabel("Photon number n̄")
        ax.set_title(_calibration_title(fit_q, name))
        ax.legend()

    grid.fig.suptitle("AC Stark photon calibration (n̄ against tone power; red = carried to the readout frequency)")
    grid.fig.set_size_inches(15, 9)
    _enlarge_fonts(grid.fig)
    grid.fig.tight_layout()
    return grid.fig


def plot_dephasing_vs_photon_number(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the induced dephasing against the photon number, with the line physics predicts.

    The predicted slope contains neither the photon number nor the drive amplitude, only chi, kappa
    and the tone detuning, so the comparison is independent of the calibration itself.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        fit_q = fits.sel(qubit=name)
        n_bar = np.asarray(fit_q["n_bar"].values, dtype=float)
        gamma_d = np.asarray(fit_q["gamma_d"].values, dtype=float)
        finite = np.isfinite(n_bar) & np.isfinite(gamma_d)
        ax.plot(n_bar[finite], 1e-6 * gamma_d[finite], ls="", marker="o", ms=4, color="C0", label="measured")

        expected = float(fit_q["expected_gamma_ratio"].values)
        chi_rad = 2 * np.pi * float(fit_q["chi_hz"].values)
        if np.isfinite(expected) and finite.sum() >= 2:
            line_n = np.linspace(0, float(np.nanmax(n_bar[finite])), 50)
            ax.plot(
                line_n,
                1e-6 * expected * 2 * abs(chi_rad) * line_n,
                ls="--",
                lw=1.2,
                color="C3",
                label="predicted from χ, κ and the detuning",
            )
        ax.set_xlabel("Photon number n̄ at the tone")
        ax.set_ylabel("$\\Gamma_d$ [MHz]")
        ax.set_title(_dephasing_title(fit_q, name))
        ax.legend()

    grid.fig.suptitle("AC Stark photon calibration (induced dephasing against n̄)")
    grid.fig.set_size_inches(15, 9)
    _enlarge_fonts(grid.fig)
    grid.fig.tight_layout()
    return grid.fig


def _enlarge_fonts(fig: Figure) -> None:
    """Readable text on the 15 x 9 inch figures."""
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(15)
    for ax in fig.axes:
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=11)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(10)


def _calibration_title(fit: xr.Dataset, qubit_name: str) -> str:
    if not bool(fit["success"].values):
        return f"{qubit_name} — fit failed"
    return (
        f"{qubit_name}\n"
        f"{float(fit['photons_per_mw'].values):.3g} photons/mW at readout "
        f"(×{float(fit['readout_conversion_factor'].values):.1f} from tone)\n"
        f"n̄ = {float(fit['n_bar_at_operating_amplitude'].values):.2f} at the operating amplitude"
    )


def _dephasing_title(fit: xr.Dataset, qubit_name: str) -> str:
    if not bool(fit["success"].values):
        return f"{qubit_name} — fit failed"
    return (
        f"{qubit_name}\n"
        f"$\\Gamma_d/\\Delta\\omega$ = {float(fit['gamma_d_over_delta_omega'].values):.3f} "
        f"against a predicted {float(fit['expected_gamma_ratio'].values):.3f}"
    )
