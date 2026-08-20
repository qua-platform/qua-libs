"""Plotting utilities for resonator spectroscopy calibration.

Grid-size-aware variant (``_new``)
----------------------------------
The original module hard-coded a fixed figure size (15x9 in) **and** large
absolute font sizes tuned for a *single* 15x9 panel.  On a multiplexed grid
(e.g. 3x3) the figure stayed 15x9, so each panel shrank to ~5x3 in while the
fonts stayed at 24 pt -> giant text on tiny plots.

Here every figure derives a single scale factor ``s`` from the grid layout
(``grid.all_axes.shape``) and scales the figure size **and** all text/marker
sizes together.  Result:

* 1 qubit  -> ``s = 1`` -> 15x9 in with the original (gold) fonts, i.e. the
  single-qubit plot is byte-for-byte the look of the reference dataset.
* N qubits -> the total figure is bounded to ~``_TARGET_W`` x ``_TARGET_H``
  inches and the per-panel fonts shrink proportionally, so each panel keeps
  the same font-to-panel ratio as the single-qubit plot.

Everything is driven by the constants below, so the whole look can be retuned
in one place (raise ``_TARGET_W/_TARGET_H`` to grow panels, lower them to make
things more compact).
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .analysis import lorentzian_dip_linbg

u = unit(coerce_to_integer=True)


def _relock_twin_axis(ax: Axes, ax2: Axes, center_hz: float) -> None:
    """Re-pin the detuning twin axis ``ax2`` to the affine image of RF axis ``ax``.

    The amplitude/phase panels draw the data on a bottom RF axis ``ax`` AND a ``twiny()``
    top detuning axis ``ax2`` (related by full_freq = detuning + center_hz). They overlap
    only while the two x-axes stay affine-aligned; on a bad/edge fit the FWHM band
    ``axvspan`` (or any marker) outside the swept range independently stretches ``ax2``'s
    autoscale, desyncing the twins so the one trace renders as "two diverging lines". Call
    this AFTER all overlays to force ``ax2`` back onto ``ax`` (a no-op when already aligned).
    """
    lo, hi = ax.get_xlim()  # GHz
    ax2.set_xlim((lo * u.GHz - center_hz) / u.MHz, (hi * u.GHz - center_hz) / u.MHz)


# ---------------------------------------------------------------------------
# Reference single-panel geometry + bounds (tune the figure look here)
# ---------------------------------------------------------------------------
# The "gold" single-qubit panel: a 1x1 grid renders at exactly this size with
# the base font sizes below, reproducing the original single-qubit figure.
_REF_PANEL_W = 15.0  # inches, width of one reference panel
_REF_PANEL_H = 9.0  # inches, height of one reference panel

# Maximum total figure size for a multiplexed grid.  The scale factor never
# exceeds 1, so panels are never *larger* than the reference; for large grids
# the figure is capped at these dimensions and fonts shrink to fit.
_TARGET_W = 24.0  # inches, max total figure width
_TARGET_H = 15.0  # inches, max total figure height

# ---------------------------------------------------------------------------
# Base (gold) font / pad sizes — these apply unscaled to a single panel
# ---------------------------------------------------------------------------
_BASE_FS_SUPTITLE = 28  # figure suptitle
_BASE_FS_TITLE = 24  # per-subplot title
_BASE_FS_LABEL = 24  # axis label (xlabel / ylabel)
_BASE_FS_TICK = 22  # tick-label size
_BASE_FS_LEGEND = 20  # legend text
_BASE_FS_CBAR = 20  # colorbar label / ticks

# Extra vertical padding (pts) for subplot titles on plots that have a twiny()
# top x-axis: the top ticks + "Detuning [MHz]" label need to be cleared.
# Rule of thumb: ~1.5 x (_FS_TICK + _FS_LABEL) in points (at full scale).
_BASE_TITLE_PAD_TWINY = 40  # plots with twiny top x-axis
_BASE_TITLE_PAD = 8  # plots without a top x-axis


def _clip(value: float, floor: float) -> float:
    """Return *value* but never below *floor* (keeps thin lines/markers visible)."""
    return value if value >= floor else floor


def _style_for_grid(grid: QubitGrid) -> SimpleNamespace:
    """Derive a grid-size-aware style from a constructed :class:`QubitGrid`.

    ``grid.all_axes`` is the full ``(nrows, ncols)`` array of axes created by
    ``plt.subplots`` inside ``QubitGrid``, so its shape gives the layout.  The
    scale factor ``s`` is chosen so the total figure fits within
    ``_TARGET_W`` x ``_TARGET_H`` while never exceeding the reference panel
    size (``s <= 1``).  All font / pad / line / marker sizes scale with ``s``.
    """
    nrows, ncols = grid.all_axes.shape
    s = min(
        1.0,
        _TARGET_W / (_REF_PANEL_W * ncols),
        _TARGET_H / (_REF_PANEL_H * nrows),
    )
    return SimpleNamespace(
        s=s,
        nrows=nrows,
        ncols=ncols,
        # Total figure size (bounded by the targets; >= a single reference panel)
        fig_w=_REF_PANEL_W * s * ncols,
        fig_h=_REF_PANEL_H * s * nrows,
        # Fonts scale linearly with the panel
        fs_suptitle=_BASE_FS_SUPTITLE * s,
        fs_title=_BASE_FS_TITLE * s,
        fs_label=_BASE_FS_LABEL * s,
        fs_tick=_BASE_FS_TICK * s,
        fs_legend=_BASE_FS_LEGEND * s,
        fs_cbar=_BASE_FS_CBAR * s,
        # Title padding scales with the panel too (it must clear the top axis)
        pad_twiny=_BASE_TITLE_PAD_TWINY * s,
        pad=_BASE_TITLE_PAD * s,
        # Line widths scale linearly (with small visibility floors)
        lw_fit=_clip(1.5 * s, 0.8),
        lw_gd=_clip(1.0 * s, 0.6),
        lw_line=_clip(1.0 * s, 0.6),
        lw_thin=_clip(0.8 * s, 0.5),
        lw_hair=_clip(0.5 * s, 0.4),
        # Marker areas: small dots scale ~linearly, the big f0 star ~by area
        scatter_s=_clip(8.0 * s, 3.0),
        star_s=_clip(200.0 * s * s, 40.0),
        arrow_scale=_clip(14.0 * s, 8.0),
    )


def _apply_tick_fontsize(ax: Axes, size: float) -> None:
    """Set tick-label fontsize on both axes of *ax*."""
    ax.tick_params(axis="both", labelsize=size)


def _finalize(grid: QubitGrid, st: SimpleNamespace, suptitle: str) -> Figure:
    """Apply the shared figure-level styling (size, suptitle, layout)."""
    grid.fig.suptitle(suptitle, fontsize=st.fs_suptitle)
    grid.fig.set_size_inches(st.fig_w, st.fig_h)
    # Reserve a little headroom for the (scaled) suptitle so it never collides
    # with the per-subplot titles / twiny top axes.
    grid.fig.tight_layout(rect=[0, 0, 1, 0.97])
    return grid.fig


def plot_raw_phase(
    ds: xr.Dataset,
    qubits: list[AnyTransmon],
    fits: xr.Dataset | None = None,
) -> Figure:
    """Plot raw phase with group delay overlay and optional f0 marker.

    Parameters
    ----------
    ds:
        Raw dataset containing 'phase', 'detuning', and 'full_freq'.
    qubits:
        List of transmon qubits.
    fits:
        Optional fit dataset (ds_fit).  When provided, a vertical dashed red
        line is drawn at the fitted resonance frequency f0.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    for ax1, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        full_freq_ghz = ds_q.full_freq.values / u.GHz
        full_freq_hz = ds_q.full_freq.values
        phase = ds_q.phase.values
        qubit_name = qubit["qubit"]

        # Primary axis: absolute RF frequency [GHz], left y: phase
        ds_q.assign_coords(full_freq_GHz=ds_q.full_freq / u.GHz).phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]", fontsize=st.fs_label)
        ax1.set_ylabel("phase [rad]", fontsize=st.fs_label)
        _apply_tick_fontsize(ax1, st.fs_tick)

        # Top x-axis: detuning [MHz]
        ax2 = ax1.twiny()
        ds_q.assign_coords(detuning_MHz=ds_q.detuning / u.MHz).phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
        ax2.set_title("")  # clear any auto-title xarray set on the twin axis
        _apply_tick_fontsize(ax2, st.fs_tick)

        # Right y-axis: group delay  -dφ/df  [ns]
        tau_ns = -np.gradient(phase, full_freq_hz) * 1e9
        ax_gd = ax1.twinx()
        ax_gd.plot(
            full_freq_ghz,
            tau_ns,
            color="darkorange",
            linewidth=st.lw_gd,
            alpha=0.7,
            label="group delay",
        )
        ax_gd.set_ylabel("-dφ/df [ns]", color="darkorange", fontsize=st.fs_label)
        ax_gd.tick_params(axis="y", colors="darkorange", labelsize=st.fs_tick)

        # Explicit centered subplot title — pad clears the twiny top-axis label
        ax1.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad_twiny)

        # Vertical dashed line at fitted f0
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            f0_hz = float(fit_q.f0.values)
            if not np.isnan(f0_hz):
                f0_ghz_val = f0_hz / u.GHz
                ax1.axvline(
                    f0_ghz_val,
                    color="red",
                    linestyle="--",
                    linewidth=st.lw_gd,
                    label=f"f₀={f0_ghz_val:.4f} GHz",
                )
                ax1.legend(fontsize=st.fs_legend, loc="upper right")

        # Keep the detuning twin axis pinned to the RF axis (one trace, not two).
        _relock_twin_axis(ax1, ax2, full_freq_hz[0] - ds_q.detuning.values[0])

    return _finalize(grid, st, "Resonator spectroscopy (phase)")


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: list[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot IQ amplitude with fitted curves for all qubits on a grid."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    for ax, qubit in grid_iter(grid):
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), st)
    return _finalize(grid, st, "Resonator spectroscopy (amplitude + fit)")


def plot_individual_amplitude_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, Any],
    fit: xr.Dataset | None = None,
    st: SimpleNamespace | None = None,
) -> None:
    """Plot a single qubit's amplitude trace with the fitted Lorentzian overlay.

    The fit curve is reconstructed from the stored popt = [f0, fwhm, amp, bg0, bg1]
    using lorentzian_dip_linbg evaluated over the full absolute-frequency axis.
    A NaN popt (failed fit) suppresses the overlay.

    Enhancements:
    - Fit line is dashed (r--) instead of solid.
    - Legend shows fitted f0 [GHz] and FWHM [MHz].
    - A semi-transparent red band marks the FWHM width around f0.

    ``st`` carries the grid-size-aware font/line sizes.  It is optional so the
    helper can be reused standalone; when omitted a single-panel (gold) style
    is used.
    """
    if st is None:
        st = SimpleNamespace(
            fs_label=_BASE_FS_LABEL,
            fs_title=_BASE_FS_TITLE,
            fs_tick=_BASE_FS_TICK,
            fs_legend=_BASE_FS_LEGEND,
            pad_twiny=_BASE_TITLE_PAD_TWINY,
            lw_fit=1.5,
        )

    ds_q = ds.loc[qubit]
    qubit_name = qubit["qubit"]

    # Primary x-axis: absolute RF frequency in GHz
    (ds_q.assign_coords(full_freq_GHz=ds_q.full_freq / u.GHz).IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]", fontsize=st.fs_label)
    ax.set_ylabel(r"$R=\sqrt{I^2+Q^2}$ [mV]", fontsize=st.fs_label)
    _apply_tick_fontsize(ax, st.fs_tick)

    # Secondary x-axis: detuning in MHz
    ax2 = ax.twiny()
    (ds_q.assign_coords(detuning_MHz=ds_q.detuning / u.MHz).IQ_abs / u.mV).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
    ax2.set_title("")  # clear xarray auto-title on twin axis
    _apply_tick_fontsize(ax2, st.fs_tick)

    # Explicit centered subplot title — three-state in v2:
    #   black   "qubit = qA1"                  frequency + lineshape both good
    #   amber   "... (freq OK, shape poor)"    dip found; Lorentzian gates failed
    #   crimson "... (NO DIP)"                 no significant dip — nothing written
    # An "[AMBIG]" suffix marks windows with several comparable dips
    # (downstream disambiguates via expected frequency / punch-out).
    title = f"qubit = {qubit_name}"
    color = "black"
    if fit is not None and "success" in fit:
        _succ = bool(fit.success.values)
        _shape = bool(fit.success_shape.values) if "success_shape" in fit else _succ
        if not _succ:
            title += "  (NO DIP)"
            color = "crimson"
        elif not _shape:
            title += "  (freq OK, shape poor)"
            color = "darkorange"
        if "ambiguous" in fit and bool(fit.ambiguous.values):
            title += "  [AMBIG]"
            if color == "black":
                color = "darkorange"
    ax.set_title(title, loc="center", fontsize=st.fs_title, pad=st.pad_twiny, color=color)

    # Overlay fitted curve if the fit succeeded (popt has no NaN)
    if fit is not None:
        popt = fit.popt.values  # shape (5,): [f0, fwhm, amp, bg0, bg1]
        if not np.any(np.isnan(popt)):
            full_freq_q = ds_q.full_freq.values  # absolute Hz
            detuning_mhz = ds_q.detuning.values / u.MHz

            # Restrict the drawn curve to the window popt was actually fit on: the
            # background (bg0, bg1) is only valid there, and linearly extrapolating
            # it across the full sweep can diverge sharply from the data outside it.
            if "fit_win_lo" in fit and "fit_win_hi" in fit:
                win_lo, win_hi = float(fit.fit_win_lo.values), float(fit.fit_win_hi.values)
                if not (np.isnan(win_lo) or np.isnan(win_hi)):
                    win_mask = (full_freq_q >= win_lo) & (full_freq_q <= win_hi)
                    full_freq_q = full_freq_q[win_mask]
                    detuning_mhz = detuning_mhz[win_mask]

            fitted_curve_mv = lorentzian_dip_linbg(full_freq_q, *popt) / u.mV

            f0_ghz = popt[0] / u.GHz
            fwhm_mhz = popt[1] / u.MHz

            # Dashed fit line with annotation in legend
            ax2.plot(
                detuning_mhz,
                fitted_curve_mv,
                "r--",
                linewidth=st.lw_fit,
                label=f"fit: f₀={f0_ghz:.4f} GHz, FWHM={fwhm_mhz:.2f} MHz",
            )

            # Semi-transparent FWHM band (on detuning axis). center_hz is the constant
            # RF_frequency offset (full_freq - detuning); must come from the unfiltered
            # arrays so its index-0 pairing is valid even after full_freq_q is windowed.
            center_hz = ds_q.full_freq.values[0] - ds_q.detuning.values[0]
            det_f0_mhz = (popt[0] - center_hz) / u.MHz
            ax2.axvspan(
                det_f0_mhz - fwhm_mhz / 2,
                det_f0_mhz + fwhm_mhz / 2,
                alpha=0.15,
                color="red",
            )

            ax2.legend(fontsize=st.fs_legend, loc="upper right")

    # Pin the detuning twin to the RF axis after all overlays (fixes the "two diverging
    # lines" desync when a bad/edge fit's FWHM band falls outside the swept range).
    _relock_twin_axis(ax, ax2, ds_q.full_freq.values[0] - ds_q.detuning.values[0])


def plot_detrended_phase(
    ds: xr.Dataset,
    qubits: list[AnyTransmon],
    fits: xr.Dataset | None = None,
) -> Figure:
    """Plot background-subtracted phase to isolate the resonance phase step.

    A degree-3 polynomial is fitted to phase data outside ±3×FWHM around f0
    (when fit data is available) and subtracted from the raw phase.  Without
    fit data a global degree-3 polynomial is used.  This removes the large
    bowl-shaped cable/electronics background and shows just the dispersive
    resonance phase step.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)
    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        detuning_mhz = ds_q.detuning.values / u.MHz
        phase = ds_q.phase.values
        full_freq_hz = ds_q.full_freq.values
        qubit_name = qubit["qubit"]

        det_f0_mhz = None
        f0_hz_val = np.nan
        bg_mask = np.ones(len(detuning_mhz), dtype=bool)

        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            f0_hz_val = float(fit_q.f0.values)
            fwhm_hz_val = float(fit_q.fwhm.values)
            if not np.isnan(f0_hz_val) and not np.isnan(fwhm_hz_val) and fwhm_hz_val > 0:
                center_hz = full_freq_hz[0] - ds_q.detuning.values[0]
                det_f0_mhz = (f0_hz_val - center_hz) / u.MHz
                exclusion_half_mhz = 3.0 * fwhm_hz_val / u.MHz
                bg_mask = np.abs(detuning_mhz - det_f0_mhz) > exclusion_half_mhz

        # Polynomial background fit; fall back to full trace if mask too sparse
        if bg_mask.sum() >= 4:
            coeffs = np.polyfit(detuning_mhz[bg_mask], phase[bg_mask], deg=3)
        else:
            coeffs = np.polyfit(detuning_mhz, phase, deg=3)
        phase_bg = np.polyval(coeffs, detuning_mhz)
        phase_detrended = phase - phase_bg

        ax.plot(detuning_mhz, phase_detrended, color="steelblue", linewidth=st.lw_line)
        ax.axhline(0, color="gray", linewidth=st.lw_hair, linestyle=":")
        ax.set_xlabel("Detuning [MHz]", fontsize=st.fs_label)
        ax.set_ylabel("phase residual [rad]", fontsize=st.fs_label)
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)
        _apply_tick_fontsize(ax, st.fs_tick)

        if det_f0_mhz is not None:
            ax.axvline(
                det_f0_mhz,
                color="red",
                linestyle="--",
                linewidth=st.lw_gd,
                label=f"f₀={f0_hz_val / u.GHz:.4f} GHz",
            )
            ax.legend(fontsize=st.fs_legend, loc="upper right")

    return _finalize(grid, st, "Resonator spectroscopy (detrended phase)")


def plot_iq_circle(
    ds: xr.Dataset,
    qubits: list[AnyTransmon],
    fits: xr.Dataset | None = None,
) -> Figure:
    """Plot I vs Q parametric trace, colour-coded by detuning.

    Each point corresponds to one frequency step.  A semi-transparent line
    connects the points in frequency order to make the trace direction visible.
    Evenly-spaced arrows are added along the trace to indicate sweep direction.
    The colour encodes the detuning value so that the rotation around the
    resonance is immediately visible.  A red star marks the IQ point closest
    to the fitted f0.

    A single shared colorbar is used for the whole figure so that per-subplot
    axes are not shifted — this keeps subplot titles visually centered.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    st = _style_for_grid(grid)

    for ax, qubit in grid_iter(grid):
        ds_q = ds.loc[qubit]
        I_mV = ds_q["I"].values / u.mV
        Q_mV = ds_q["Q"].values / u.mV
        detuning_mhz = ds_q.detuning.values / u.MHz
        qubit_name = qubit["qubit"]

        # Connecting line (behind dots) to show trace continuity
        ax.plot(I_mV, Q_mV, color="gray", linewidth=st.lw_thin, alpha=0.35, zorder=1)

        # Scatter coloured by detuning
        sc = ax.scatter(
            I_mV,
            Q_mV,
            c=detuning_mhz,
            cmap="plasma",
            s=st.scatter_s,
            zorder=2,
        )

        # Directional arrows evenly spaced along the trace (skip for a degenerate trace)
        n_arrows = 8
        arrow_indices = np.linspace(0, len(I_mV) - 2, n_arrows, dtype=int) if len(I_mV) >= 3 else []
        for ai in arrow_indices:
            ax.annotate(
                "",
                xy=(I_mV[ai + 1], Q_mV[ai + 1]),
                xytext=(I_mV[ai], Q_mV[ai]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="dimgray",
                    lw=st.lw_fit,
                    alpha=0.55,
                    mutation_scale=st.arrow_scale,
                ),
                zorder=4,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("I [mV]", fontsize=st.fs_label)
        ax.set_ylabel("Q [mV]", fontsize=st.fs_label)
        _apply_tick_fontsize(ax, st.fs_tick)

        # Per-subplot colorbar (placed adjacent to this axes)
        cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Detuning [MHz]", fontsize=st.fs_cbar)
        cbar.ax.tick_params(labelsize=st.fs_cbar)

        # Title set AFTER colorbar so it is centered over the (now-final) axes width
        ax.set_title(f"qubit = {qubit_name}", loc="center", fontsize=st.fs_title, pad=st.pad)

        # Mark the IQ point nearest the fitted resonance frequency
        if fits is not None:
            fit_q = fits.sel(qubit=qubit_name)
            f0_hz = float(fit_q.f0.values)
            if not np.isnan(f0_hz):
                full_freq_hz = ds_q.full_freq.values
                idx = int(np.argmin(np.abs(full_freq_hz - f0_hz)))
                ax.scatter(
                    I_mV[idx],
                    Q_mV[idx],
                    color="red",
                    marker="*",
                    s=st.star_s,
                    zorder=5,
                    label=f"f₀={f0_hz / u.GHz:.4f} GHz",
                )
                ax.legend(fontsize=st.fs_legend, loc="upper right")

    return _finalize(grid, st, "Resonator spectroscopy (IQ circle)")
