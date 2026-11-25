from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _drive_amplitude(ds: xr.Dataset, target: str, drive: str) -> xr.DataArray:
    """Amplitude played on the drive line, in output (DAC) units."""
    return (
        ds.amp_prefactor
        * ds.drive_base_amplitude.sel(drive_qubit=drive)
        * ds.pair_amp_scale.sel(qubit=target, drive_qubit=drive)
    )


def _finalize(grid: QubitGrid, title: str):
    grid.fig.suptitle(title)
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_amplitude_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the power Rabi of every drive line, measured on each target qubit, with the fit.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw amplitude-sweep data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    fits : xr.Dataset
        Dataset containing the fit results.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots, one panel per target qubit.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_amplitude_data_with_fit(ax, ds, qubit["qubit"], fits)
    return _finalize(grid, "Drive-line crosstalk: power Rabi of the target vs. drive-line amplitude")


def plot_individual_amplitude_data_with_fit(ax: Axes, ds: xr.Dataset, target: str, fits: xr.Dataset):
    """Plot the power Rabi of every drive line on a single target qubit."""
    for drive in ds.drive_qubit.values:
        if not bool(ds.pair_valid.sel(qubit=target, drive_qubit=drive)):
            continue
        x = _drive_amplitude(ds, target, drive)
        ratio = float(fits.amplitude_ratio.sel(qubit=target, drive_qubit=drive))
        label = f"drive {drive}" if drive == target else f"drive {drive}, r={ratio:.4f}"
        (line,) = ax.plot(x, ds.state.sel(qubit=target, drive_qubit=drive), ".", ms=3, label=label)
        fit = fits.fit.sel(qubit=target, drive_qubit=drive)
        if np.all(np.isfinite(fit)):
            y = oscillation(ds.amp_prefactor.values, *[float(fit.sel(fit_vals=k)) for k in ("a", "f", "phi", "offset")])
            ax.plot(x, y, color=line.get_color(), lw=1)
    ax.set_xlabel("Drive-line amplitude [a.u.]")
    ax.set_ylabel("Excited-state population")
    ax.set_title(f"Target {target}")
    ax.legend(fontsize=7)


def plot_phase_data(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the mean target excitation vs. compensation phase for the coarse and fine sweeps of every pair.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots, one panel per target qubit.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_phase_data(ax, ds, qubit["qubit"], fits)
    return _finalize(grid, "Drive-line crosstalk: target excitation vs. compensation phase")


def plot_individual_phase_data(ax: Axes, ds: xr.Dataset, target: str, fits: xr.Dataset):
    """Plot the compensation-phase sweeps of every drive line on a single target qubit."""
    for drive in ds.drive_qubit.values:
        selection = dict(qubit=target, drive_qubit=drive)
        if drive == target or not np.all(np.isfinite(ds.state.sel(selection))):
            continue
        cost = ds.state.sel(selection).mean("amp_prefactor")
        phases = np.rad2deg(2 * np.pi * (ds.phase_turns.sel(selection) % 1))
        (line,) = ax.plot(phases.sel(phase_sweep="coarse"), cost.sel(phase_sweep="coarse"), ".-", ms=3, lw=0.5)
        ax.plot(phases.sel(phase_sweep="fine"), cost.sel(phase_sweep="fine"), "o", ms=3, color=line.get_color())
        theta = np.rad2deg(float(fits.fitted_phase.sel(selection)))
        ax.axvline(theta, color=line.get_color(), ls="--", label=f"drive {drive}: {theta:.1f} deg")
    ax.set_xlabel("Compensation phase [deg]")
    ax.set_ylabel("Mean excited-state population")
    ax.set_xlim(0, 360)
    ax.set_title(f"Target {target}")
    ax.legend(fontsize=7)


def plot_crosstalk_matrix(fits: xr.Dataset):
    """
    Plots the crosstalk matrix as a table: one row per drive line, one column per target qubit.

    Each cell shows the amplitude ratio r and the compensation phase. Pairs that failed the compensation check, or whose
    power Rabi could not be fitted, are marked FAIL. The cell colour is log10(r).

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the table.
    """
    drives, targets = list(fits.drive_qubit.values), list(fits.qubit.values)
    ratio = fits.amplitude_ratio.transpose("drive_qubit", "qubit")
    phase = fits.fitted_phase.transpose("drive_qubit", "qubit")
    success = fits.success.transpose("drive_qubit", "qubit")
    values = np.log10(ratio.values.astype(float))
    for i, drive in enumerate(drives):
        for j, target in enumerate(targets):
            if drive == target:
                values[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(2.2 * len(targets) + 2, 1.3 * len(drives) + 1.5))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="RdPu", vmin=-3, vmax=0, aspect="auto")
    for i, drive in enumerate(drives):
        for j, target in enumerate(targets):
            r, theta = float(ratio.values[i, j]), float(phase.values[i, j])
            if drive == target:
                label = "—"
            elif not np.isfinite(r):
                label = "FAIL"
            else:
                theta_label = f"{np.rad2deg(theta) % 360:.1f}°" if np.isfinite(theta) else "n/a"
                label = f"r = {r:.4f}\nθ = {theta_label}"
                if not bool(success.values[i, j]):
                    label += "\nFAIL"
            dark = np.isfinite(values[i, j]) and values[i, j] > -1.2
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="white" if dark else "black")
    ax.set_xticks(range(len(targets)), targets)
    ax.set_yticks(range(len(drives)), [f"drive {d}" for d in drives])
    ax.set_xlabel("Target qubit")
    ax.set_ylabel("Drive line")
    ax.set_title("Drive-line crosstalk matrix: amplitude ratio r and compensation phase θ")
    fig.colorbar(image, ax=ax, label="log10(r)")
    fig.tight_layout()
    return fig


def plot_check_data(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the no-drive, uncompensated and compensated target excitation of every pair.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots, one panel per target qubit.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_check_data(ax, ds, qubit["qubit"], fits)
    return _finalize(grid, "Drive-line crosstalk: compensation check (dashed: uncompensated, solid: compensated)")


def plot_individual_check_data(ax: Axes, ds: xr.Dataset, target: str, fits: xr.Dataset):
    """Plot the compensation check of every drive line on a single target qubit."""
    for drive in ds.drive_qubit.values:
        selection = dict(qubit=target, drive_qubit=drive)
        if drive == target or not np.all(np.isfinite(ds.state.sel(selection))):
            continue
        x = _drive_amplitude(ds, target, drive)
        verdict = "pass" if bool(fits.success.sel(selection)) else "fail"
        state = ds.state.sel(selection)
        (line,) = ax.plot(x, state.sel(mode="uncompensated"), "--", lw=1)
        ax.plot(x, state.sel(mode="compensated"), "-", color=line.get_color(), label=f"drive {drive}: {verdict}")
        ax.plot(x, state.sel(mode="no_drive"), ":", color="gray", lw=1)
    ax.set_xlabel("Drive-line amplitude [a.u.]")
    ax.set_ylabel("Excited-state population")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Target {target}")
    ax.legend(fontsize=7)
