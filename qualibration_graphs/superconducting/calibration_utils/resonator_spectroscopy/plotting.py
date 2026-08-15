from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot raw phase data for the given qubits."""

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        ax2 = ax1.twiny()
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot resonator spectroscopy amplitude with the fitted Lorentzian overlay."""

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot one qubit's amplitude trace and fitted curve on the primary frequency axis."""

    qubit_ds = ds.loc[qubit]
    x_ghz = qubit_ds.full_freq / u.GHz
    y_mv = qubit_ds.IQ_abs / u.mV
    ax.plot(x_ghz, y_mv, color="C0", label="data")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")

    rf_hz = float(qubit_ds.full_freq.isel(detuning=0) - qubit_ds.detuning.isel(detuning=0))
    if fit is not None and "fit_curve" in fit:
        fit_y_mv = fit.fit_curve / u.mV
        ax.plot(x_ghz, fit_y_mv, "r--", label="fit")
        center_ghz = float(fit.res_freq / u.GHz)
        ax.axvline(center_ghz, color="tab:orange", linestyle=":", linewidth=1.2, label="fit center")
        feature = "peak" if bool(fit.feature_is_peak.values) else "dip"
        ax.text(
            0.02,
            0.98,
            f"{feature} | R²={float(fit.fit_r2.values):.3f}\nRMSE={1e3 * float(fit.fit_rmse.values):.2f} mV",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        ax.legend(loc="lower right", fontsize=8)

    ax2 = ax.twiny()
    f0, f1 = ax.get_xlim()
    ax2.set_xlim((f0 * u.GHz - rf_hz) / u.MHz, (f1 * u.GHz - rf_hz) / u.MHz)
    ax2.set_xlabel("Detuning [MHz]")
