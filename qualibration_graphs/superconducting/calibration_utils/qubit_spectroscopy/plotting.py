"""Plotting utilities for qubit spectroscopy calibration (v2).

Per-qubit panel keeps the dual-axis (absolute RF + detuning) layout from the
legacy plot but overlays the explicit Lorentzian+linear-bg fit curve (from
the stored `popt`) and an in-panel values box listing every fit-derived
value that `update_state` may write to QUAM: f₀, FWHM, R², contrast,
integration-weight angle, saturation amplitude, x180 amplitude.

Failed-fit panels get a red box border and a `(FAILED)` suffix on the title
so bad panels are obvious at a glance.
"""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import lorentzian_peak_linbg

u = unit(coerce_to_integer=True)


_FS_SUPTITLE = 22
_FS_TITLE = 16
_FS_LABEL = 14
_FS_TICK = 12
_FS_LEGEND = 11
_FS_BOX = 10

_TITLE_PAD = 6

# Per-panel sizing. Width per column accommodates the dual axis labels +
# values box; height per row leaves space for title and ticks.
_PER_COL_INCH = 6.0
_PER_ROW_INCH = 4.5
_MIN_WIDTH_INCH = 8.0
_MIN_HEIGHT_INCH = 6.0


def _setup_grid_figure(grid) -> None:
    """Resize the QubitGrid figure to its panel count and apply tight_layout.

    constrained_layout was tried but it crashes (ZeroDivisionError in
    matplotlib's _layoutgrid) when combined with `ax.axis("off")` cells
    that QubitGrid uses for empty grid slots and `bbox_inches="tight"` that
    qualang_tools' data handler applies at save time.
    """
    nrows, ncols = grid.all_axes.shape
    width = max(_MIN_WIDTH_INCH, ncols * _PER_COL_INCH)
    height = max(_MIN_HEIGHT_INCH, nrows * _PER_ROW_INCH)
    grid.fig.set_size_inches(width, height)
    grid.fig.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)


def _apply_tick_fontsize(ax: Axes, size: int = _FS_TICK) -> None:
    ax.tick_params(axis="both", labelsize=size)


def _format_values_box(fit: xr.Dataset, update_pulses: bool) -> str:
    """Build the per-panel multi-line values text box.

    Shows only the values that get written back to QUAM (or the
    user-facing measurements). R² and contrast were intentionally removed
    per user feedback — they are still computed in `analysis.py` and stored
    in `ds_fit` / `data.json`; the figure stays uncluttered while the
    failure indicator (red border + (FAILED) title) communicates the gate
    result without needing the underlying metric on the panel.
    """
    f0_hz = float(fit.res_freq.values)
    fwhm_hz = float(fit.fwhm.values)
    iw = float(fit.iw_angle.values)
    sat_amp = float(fit.saturation_amplitude.values)
    x180_amp = float(fit.x180_amplitude.values)

    lines = [
        f"f₀       = {f0_hz / u.GHz:.5f} GHz",
        f"FWHM    = {fwhm_hz / u.MHz:.3f} MHz",
        f"iw angle = {iw:+.3f} rad",
    ]
    if update_pulses:
        lines.append(f"sat amp  = {sat_amp * 1e3:.2f} mV")
        lines.append(f"x180 amp = {x180_amp * 1e3:.2f} mV")
    return "\n".join(lines)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Per-qubit rotated-I vs detuning panel with Lorentzian fit overlay + values box."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    update_pulses = _detect_update_pulses_flag(fits)
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(
            ax,
            ds,
            qubit,
            fits.sel(qubit=qubit["qubit"]),
            update_pulses,
        )
    grid.fig.suptitle(
        "Qubit spectroscopy (rotated I + Lorentzian fit)",
        fontsize=_FS_SUPTITLE,
    )
    _setup_grid_figure(grid)
    return grid.fig


def _detect_update_pulses_flag(fits: xr.Dataset) -> bool:
    """True iff every qubit has a finite saturation_amplitude (proxy for the param)."""
    if "saturation_amplitude" not in fits:
        return False
    vals = fits.saturation_amplitude.values
    return bool(np.all(np.isfinite(vals)))


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset,
    update_pulses: bool,
) -> None:
    """Single-qubit panel with fit overlay, dual axes, and a fit-values text box."""
    ds_q = ds.sel(qubit=qubit["qubit"]) if "qubit" in ds.dims else ds.loc[qubit]
    fit_q = fit  # already qubit-selected
    qubit_name = qubit["qubit"]

    detuning_hz = ds_q.detuning.values
    full_freq_hz = ds_q.full_freq.values
    i_rot_mv = fit_q.I_rot.values / u.mV if "I_rot" in fit_q else ds_q.I_rot.values / u.mV

    # Primary axis: absolute RF [GHz]
    ax.plot(full_freq_hz / u.GHz, i_rot_mv, color="steelblue", linewidth=1.0)
    ax.set_xlabel("RF frequency [GHz]", fontsize=_FS_LABEL)
    ax.set_ylabel("Rotated I [mV]", fontsize=_FS_LABEL)
    _apply_tick_fontsize(ax)

    # Secondary x-axis: detuning [MHz]
    ax2 = ax.twiny()
    ax2.set_xlim(detuning_hz[0] / u.MHz, detuning_hz[-1] / u.MHz)
    ax2.set_xlabel("Detuning [MHz]", fontsize=_FS_LABEL)
    _apply_tick_fontsize(ax2)

    success = bool(fit_q.success.values) if "success" in fit_q else False
    popt = fit_q.popt.values if "popt" in fit_q else None

    # Fit overlay — drawn only within the window the fit was actually done over,
    # so the linear-bg slope isn't visually extrapolated across the full scan
    # (the previous behaviour produced a "tilted tail" that drifted from the
    # data's flat baseline even when the fit was correct).
    if popt is not None and not np.any(np.isnan(popt)):
        f0_rel = float(fit_q.f0.values) if "f0" in fit_q else float(popt[0])
        fwhm = float(fit_q.fwhm.values) if "fwhm" in fit_q else float(popt[1])
        if "fit_window_half_hz" in fit_q and np.isfinite(float(fit_q.fit_window_half_hz.values)):
            window_half = float(fit_q.fit_window_half_hz.values)
        else:
            window_half = max(4.0 * fwhm, 10e6)
        mask = np.abs(detuning_hz - f0_rel) <= window_half
        if mask.sum() >= 4:
            fit_curve_mv = lorentzian_peak_linbg(detuning_hz[mask], *popt) / u.mV
            ax.plot(
                full_freq_hz[mask] / u.GHz,
                fit_curve_mv,
                "r--",
                linewidth=1.5,
                label="Lorentzian fit",
            )
        # f₀ marker
        f0_hz = float(fit_q.res_freq.values)
        ax.axvline(
            f0_hz / u.GHz,
            color="red",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.legend(fontsize=_FS_LEGEND, loc="upper left")

    # Title (with FAILED suffix if appropriate)
    title = qubit_name + ("  (FAILED)" if not success else "")
    ax.set_title(title, loc="center", fontsize=_FS_TITLE, pad=_TITLE_PAD, color=("crimson" if not success else "black"))

    # Values box anchored upper-right; red border on failure.
    if popt is not None and not np.any(np.isnan(popt)):
        text = _format_values_box(fit_q, update_pulses)
        edge = "crimson" if not success else "gray"
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_FS_BOX,
            family="monospace",
            bbox=dict(
                facecolor="white",
                alpha=0.88,
                edgecolor=edge,
                linewidth=1.5,
            ),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "NO FIT",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=_FS_LABEL,
            color="gray",
        )
