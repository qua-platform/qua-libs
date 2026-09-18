from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import fit_circle

u = unit(coerce_to_integer=True)

__all__ = ["plot_circle_fit", "plot_magnitude_with_fit", "plot_dispersive_shift"]


def plot_circle_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the normalised complex trace with the fitted circle on top, one panel per qubit.

    A bad fit is visible as a fitted curve that does not follow the measured points, which is the
    point of showing the two together rather than quoting a number.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_single_circle(ax, fits.sel(qubit=qubit["qubit"]), qubit["qubit"])

    grid.fig.suptitle("Resonator linewidth (normalised S21 and circle fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single_circle(ax: Axes, fit: xr.Dataset, qubit_name: str) -> None:
    measured_i = fit["S21_normalised_I"].values
    measured_q = fit["S21_normalised_Q"].values
    model_i = fit["S21_model_I"].values
    model_q = fit["S21_model_Q"].values

    ax.plot(measured_i, measured_q, ls="", marker=".", ms=4, color="C0", label="data")
    # The fitted circle is drawn smoothly rather than by joining the model at the swept frequencies,
    # which would look polygonal wherever the sweep steps quickly around the circle.
    circle = _smooth_circle(model_i, model_q)
    if circle is not None:
        ax.plot(circle[0], circle[1], ls="-", lw=1.5, color="C3", label="circle fit")
    ax.plot(model_i, model_q, ls="", marker="x", ms=3, color="C3", label="fit at the swept points")
    ax.set_aspect("equal")
    ax.set_xlabel("Re S21 (normalised)")
    ax.set_ylabel("Im S21 (normalised)")
    ax.set_title(_panel_title(fit, qubit_name))
    ax.legend(fontsize=7)


def _smooth_circle(model_i, model_q):
    """Return a densely sampled circle through the fitted model points, or None when it degenerates."""
    finite = np.isfinite(model_i) & np.isfinite(model_q)
    if finite.sum() < 4:
        return None
    try:
        center, radius = fit_circle(model_i[finite] + 1j * model_q[finite])
    except (ValueError, np.linalg.LinAlgError):
        return None
    angle = np.linspace(0, 2 * np.pi, 400)
    return center.real + radius * np.cos(angle), center.imag + radius * np.sin(angle)


def plot_magnitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot |S21| against detuning with the fitted resonance on top, one panel per qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = fits.sel(qubit=qubit["qubit"])
        detuning_mhz = fit.detuning.values / u.MHz
        measured = np.abs(fit["S21_normalised_I"].values + 1j * fit["S21_normalised_Q"].values)
        model = np.abs(fit["S21_model_I"].values + 1j * fit["S21_model_Q"].values)
        ax.plot(detuning_mhz, measured, ls="", marker=".", ms=4, color="C0", label="data")
        ax.plot(detuning_mhz, model, ls="-", lw=1.5, color="C3", label="fit")
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("|S21| (normalised)")
        ax.set_title(_panel_title(fit, qubit["qubit"]))
        ax.legend(fontsize=7)

    grid.fig.suptitle("Resonator linewidth (normalised |S21| and fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_dispersive_shift(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot |S21| with the qubit in |0> and in |1>, with both fitted resonances marked.

    The gap between the two dashed lines is 2 chi, so this figure is where chi can be read off and a
    bad splitting seen rather than inferred from a number. Returns None when the excited state was
    not measured.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        ds_q = ds.sel(qubit=name)
        fit = fits.sel(qubit=name)
        detuning_mhz = ds_q.detuning.values / u.MHz
        for state, colour, label in ((0, "C0", "|0⟩"), (1, "C3", "|1⟩")):
            magnitude = np.abs(ds_q.I.isel(state=state).values + 1j * ds_q.Q.isel(state=state).values)
            ax.plot(detuning_mhz, magnitude, ls="-", lw=1.2, marker=".", ms=3, color=colour, label=label)
        f_r = float(fit["resonance_frequency"].values)
        f_r_e = float(fit["resonance_frequency_excited"].values)
        centre = float(ds_q.full_freq.values[0]) - float(ds_q.detuning.values[0])
        for frequency, colour in ((f_r, "C0"), (f_r_e, "C3")):
            if np.isfinite(frequency):
                ax.axvline((frequency - centre) / u.MHz, color=colour, lw=0.8, ls="--")
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("|S21| [V]")
        ax.set_title(_dispersive_title(fit, name))
        ax.legend(fontsize=7)

    grid.fig.suptitle("Resonator linewidth (dispersive shift between |0⟩ and |1⟩)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _dispersive_title(fit: xr.Dataset, qubit_name: str) -> str:
    chi_mhz = float(fit["chi_hz"].values) / u.MHz
    ratio = float(fit["chi_over_kappa"].values)
    if not np.isfinite(chi_mhz):
        return f"{qubit_name} — no dispersive shift fitted"
    warning = "" if ratio < 0.3 else "  (weak-dispersive limit broken)"
    return f"{qubit_name}\n$\\chi$ = {chi_mhz:+.4f} MHz, $|\\chi|/\\kappa$ = {ratio:.2f}{warning}"


def _panel_title(fit: xr.Dataset, qubit_name: str) -> str:
    """Title carrying kappa and the probe power the internal Q was measured at."""
    if not bool(fit["success"].values):
        return f"{qubit_name} — fit failed"
    kappa_ext = float(fit["kappa_ext_hz"].values) / u.MHz
    kappa_int = float(fit["kappa_int_hz"].values) / u.MHz
    ratio = float(fit["kappa_ext_over_kappa_tot"].values)
    power = float(fit["probe_power_dbm"].values)
    return (
        f"{qubit_name}\n"
        f"$\\kappa_{{ext}}$ = {kappa_ext:.3f} MHz, $\\kappa_{{int}}$ = {kappa_int:.3f} MHz\n"
        f"$\\kappa_{{ext}}/\\kappa$ = {ratio:.2f} at {power:.1f} dBm"
    )
