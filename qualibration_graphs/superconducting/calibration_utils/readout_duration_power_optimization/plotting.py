"""Plotting utilities for the joint readout duration x power optimization.

Three figures, all on the standard per-qubit :class:`QubitGrid`:

* ``plot_fidelity_map`` -- the assignment fidelity over (integration duration, amplitude
  prefactor), with the chosen operating point marked and points rejected by the blob
  quality gates hatched out, so a gate that ate the best-looking region is visible rather
  than implied.
* ``plot_amplitude_cut`` -- the node 08b picture, taken as a slice at the chosen duration.
* ``plot_duration_cut`` -- fidelity against integration duration at the chosen amplitude,
  which is where the saturation knee shows up.

The IQ blobs and the confusion matrix at the operating point are plotted by the existing
``calibration_utils.iq_blobs`` helpers, which the node calls directly.
"""

from typing import List

import matplotlib.patheffects as pe
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# High-contrast overlays: the colormap's mid-tones swallow thin plain-coloured lines.
_HALO = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]
_OPT_COLOR = "#FF8000"


def _base_amplitude(fit: xr.Dataset) -> float:
    """Absolute readout amplitude at prefactor 1 (constant across the sweep)."""
    return float((fit.readout_amplitude / fit.amp_prefactor).median())


def _eligible(fit: xr.Dataset, outliers_threshold: float, max_variance_ratio: float) -> xr.DataArray:
    return (fit.fit_data.sel(fit_vals="outliers") >= outliers_threshold) & (
        fit.fit_data.sel(fit_vals="variance_ratio") <= max_variance_ratio
    )


def plot_fidelity_map(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    outliers_threshold: float,
    max_variance_ratio: float,
) -> Figure:
    """Grid of per-qubit fidelity heatmaps over integration duration and amplitude."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_fidelity_map(ax, fits.sel(qubit=qubit["qubit"]), qubit, outliers_threshold, max_variance_ratio)
    grid.fig.suptitle("Readout optimization: assignment fidelity vs duration and amplitude")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_fidelity_map(
    ax: Axes,
    fit: xr.Dataset,
    qubit: dict[str, str],
    outliers_threshold: float,
    max_variance_ratio: float,
):
    """Single-qubit fidelity heatmap with rejected points hatched and the optimum marked."""
    fidelity = fit.fit_data.sel(fit_vals="meas_fidelity")
    (100 * fidelity).plot(
        ax=ax,
        x="duration",
        y="amp_prefactor",
        add_colorbar=True,
        cbar_kwargs={"label": "assignment fidelity [%]"},
    )

    # Hatch the points the quality gates rejected, so a gate that removed the apparent
    # optimum is visible on the same picture as the optimum that was chosen instead.
    rejected = ~_eligible(fit, outliers_threshold, max_variance_ratio)
    if bool(rejected.any()):
        ax.contourf(
            fidelity.duration.values,
            fidelity.amp_prefactor.values,
            rejected.transpose("amp_prefactor", "duration").values.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["xx"],
        )

    opt_duration = float(fit.optimal_duration)
    opt_amp = float(fit.optimal_amp_prefactor)
    if np.isfinite(opt_duration) and np.isfinite(opt_amp):
        ax.axvline(opt_duration, color=_OPT_COLOR, lw=1.4, path_effects=_HALO)
        ax.axhline(opt_amp, color=_OPT_COLOR, lw=1.4, path_effects=_HALO)
        ax.plot(
            opt_duration,
            opt_amp,
            marker="o",
            color=_OPT_COLOR,
            markersize=7,
            path_effects=_HALO,
            label="operating point",
        )
        ax.legend(fontsize=7, loc="lower left")
        title = (
            f"{qubit['qubit']}  ({opt_duration:.0f} ns, x{opt_amp:.3f} = "
            f"{float(fit.optimal_amplitude) * 1e3:.1f} mV, F={100 * float(fit.optimal_fidelity):.2f}%)"
        )
    else:
        title = f"{qubit['qubit']}  (no point passed the blob quality gates)"

    ax.set_xlabel("Integration duration [ns]")
    ax.set_ylabel("Amplitude prefactor")
    base_amp = _base_amplitude(fit)
    secax = ax.secondary_yaxis("right", functions=(lambda p: p * base_amp * 1e3, lambda a: a / (base_amp * 1e3)))
    secax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(title, fontsize=9)


def plot_amplitude_cut(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Grid of per-qubit fidelity and non-outlier fraction vs amplitude, at the chosen duration."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_cut(ax, fits.sel(qubit=qubit["qubit"]), qubit, along="amp_prefactor")
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=3)
    grid.fig.suptitle("Readout optimization: amplitude cut at the chosen integration duration")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_duration_cut(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Grid of per-qubit fidelity and non-outlier fraction vs duration, at the chosen amplitude."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_cut(ax, fits.sel(qubit=qubit["qubit"]), qubit, along="duration")
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=3)
    grid.fig.suptitle("Readout optimization: duration cut at the chosen amplitude")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_cut(ax: Axes, fit: xr.Dataset, qubit: dict[str, str], along: str):
    """One-dimensional slice of the map through the operating point, along ``along``."""
    opt_duration = float(fit.optimal_duration)
    opt_amp = float(fit.optimal_amp_prefactor)
    if not (np.isfinite(opt_duration) and np.isfinite(opt_amp)):
        ax.set_title(f"{qubit['qubit']}  (no operating point)", fontsize=9)
        ax.set_xlabel("Integration duration [ns]" if along == "duration" else "Amplitude prefactor")
        return

    held = {"duration": opt_duration} if along == "amp_prefactor" else {"amp_prefactor": opt_amp}
    cut = fit.fit_data.sel(**held)
    marker = opt_amp if along == "amp_prefactor" else opt_duration

    cut.sel(fit_vals="meas_fidelity").plot(ax=ax, x=along, label="assignment fidelity")
    cut.sel(fit_vals="outliers").plot(ax=ax, x=along, label="non-outlier fraction")
    ax.axvline(marker, color="k", linestyle="dashed", label="operating point")

    ax.set_xlabel("Integration duration [ns]" if along == "duration" else "Amplitude prefactor")
    ax.set_ylabel("Fidelity / non-outlier fraction")
    ax.set_title(qubit["qubit"])
