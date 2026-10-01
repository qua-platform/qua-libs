from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

__all__ = [
    "plot_fringes",
    "plot_phase_vs_time",
    "plot_photon_calibration",
    "plot_dephasing_vs_photon_number",
]

_FLOOR_DBM = -200.0
"""Stand-in for the zero-amplitude point, whose power in dBm is minus infinity."""


def _amplitude_colours(scales: np.ndarray) -> List:
    span = float(np.ptp(scales))
    lowest = float(np.min(scales))
    normalised = (scales - lowest) / span if span > 0 else np.zeros_like(scales)
    return [plt.get_cmap("viridis")(value) for value in normalised]


def plot_fringes(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the measured fringes at the longest free evolution time, one curve per tone amplitude.

    The zero-amplitude reference is drawn in black, because every phase and every contrast in the
    analysis is measured against it. The longest time is shown because that is where the fringe is
    weakest and where the contrast floor decides what survives.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    signal_name = "state" if "state" in ds else "I"
    longest = int(np.argmax(ds.tau.values))
    tau_ns = float(ds.tau.values[longest])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        ds_q = ds.sel(qubit=name).isel(tau=longest)
        fit_q = fits.sel(qubit=name)
        scales = np.asarray(ds_q.amp_scale.values, dtype=float)
        colours = _amplitude_colours(scales)
        usable = np.asarray(fit_q["fringe_usable"].isel(tau=longest).values, dtype=float)
        for index, scale in enumerate(scales):
            n_bar = float(fit_q["n_bar"].values[index])
            label = f"a = {scale:.3f}" + ("" if not np.isfinite(n_bar) else f", n̄ = {n_bar:.3f}")
            ax.plot(
                ds_q.phase.values,
                ds_q[signal_name].isel(amp_scale=index).values,
                marker=".",
                ms=3,
                lw=1,
                ls="-" if usable[index] > 0 else ":",
                color="k" if scale == 0 else colours[index],
                label=label,
            )
        ax.set_xlabel("Phase of the second π/2 [turns]")
        ax.set_ylabel("State population" if signal_name == "state" else "I [V]")
        ax.set_title(f"{name} — τ = {tau_ns:.0f} ns")
        ax.legend(fontsize=5, ncol=2)

    grid.fig.suptitle(f"AC Stark photon calibration (fringes at τ = {tau_ns:.0f} ns; dotted = dropped)")
    grid.fig.set_size_inches(15, 9)
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
        limit_ns = float(fit_q["number_splitting_phase_limit_ns"].values)
        if np.isfinite(limit_ns) and limit_ns < taus.max():
            ax.axvspan(limit_ns, taus.max() * 1.05, color="gray", alpha=0.12, lw=0)
            ax.text(limit_ns, ax.get_ylim()[1], " 2χτ > 1 rad, dropped", fontsize=6, va="top", color="gray")
        ax.set_xlabel("Free evolution time τ [ns]")
        ax.set_ylabel("Accumulated fringe phase [rad]")
        linearity = float(fit_q["phase_linearity_r_squared"].values)
        ax.set_title(f"{name} — phase linearity R² = {linearity:.3f}")
        ax.axhline(0, color="gray", lw=0.6, ls=":")

    grid.fig.suptitle("AC Stark photon calibration (phase against free evolution time; slope is the Stark shift)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_photon_calibration(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the photon number against tone power, by both routes, with the fitted line.

    The power axis is in dBm and the photon axis is in n̄, so the figure carries both units. The
    photon number derived from the induced dephasing is drawn alongside the one from the Stark
    phase: they come from the same fringes by different routes, and two agreeing is the evidence.
    """
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

        constant = float(fit_q["photons_per_mw"].values)
        if np.isfinite(constant) and finite.sum() >= 2:
            line_dbm = np.linspace(power_dbm[finite].min(), float(fit_q["operating_power_dbm"].values), 200)
            ax.plot(line_dbm, constant * 10 ** (line_dbm / 10), ls="-", lw=1.5, color="C3", label="fitted line")

        operating_dbm = float(fit_q["operating_power_dbm"].values)
        if np.isfinite(operating_dbm):
            ax.axvline(operating_dbm, color="C3", lw=0.9, ls="--")
            ax.text(operating_dbm, ax.get_ylim()[1], " operating", fontsize=6, color="C3", va="top", rotation=90)
        single_photon_dbm = float(fit_q["single_photon_power_dbm"].values)
        if np.isfinite(single_photon_dbm):
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")

        ax.set_yscale("log")
        ax.set_xlabel("Stark tone power [dBm]")
        ax.set_ylabel("Photon number n̄")
        ax.set_title(_calibration_title(fit_q, name))
        ax.legend(fontsize=6)

    grid.fig.suptitle("AC Stark photon calibration (n̄ against tone power, extended to the operating power)")
    grid.fig.set_size_inches(15, 9)
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
        ax.set_xlabel("Photon number n̄")
        ax.set_ylabel("$\\Gamma_d$ [MHz]")
        ax.set_title(_dephasing_title(fit_q, name))
        ax.legend(fontsize=6)
        _add_power_axis(ax, float(fit_q["photons_per_mw"].values))

    grid.fig.suptitle("AC Stark photon calibration (induced dephasing against n̄)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _add_power_axis(ax: Axes, photons_per_mw: float) -> None:
    """Add a top axis showing the same points in dBm."""
    if not np.isfinite(photons_per_mw) or photons_per_mw <= 0:
        return

    def to_dbm(n_bar):
        n_bar = np.asarray(n_bar, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n_bar > 0, 10 * np.log10(np.maximum(n_bar, 1e-300) / photons_per_mw), _FLOOR_DBM)

    def to_n_bar(power_dbm):
        return photons_per_mw * 10 ** (np.asarray(power_dbm, dtype=float) / 10)

    secondary = ax.secondary_xaxis("top", functions=(to_dbm, to_n_bar))
    secondary.set_xlabel("Stark tone power [dBm]", fontsize=8)


def _calibration_title(fit: xr.Dataset, qubit_name: str) -> str:
    if not bool(fit["success"].values):
        return f"{qubit_name} — fit failed"
    return (
        f"{qubit_name}\n"
        f"{float(fit['photons_per_mw'].values):.3g} photons/mW, "
        f"n̄ = 1 at {float(fit['single_photon_power_dbm'].values):.1f} dBm\n"
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
