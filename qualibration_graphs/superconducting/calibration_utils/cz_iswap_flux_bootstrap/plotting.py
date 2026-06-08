"""Plotting for CZ / iSWAP flux bootstrap (node 30)."""

from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: list,
    fits: Optional[Dict] = None,
    analysis_debug: bool = False,
    cz_or_iswap: str = "cz",
) -> Dict[str, Figure]:
    """Plot control/target 2D maps; overlay fit markers only when values exist.

    Raw heatmaps are always drawn. Fit lines (decouple, gate point, optimal qubit
    flux) are added per pair only for finite entries in ``fits``.
    When ``analysis_debug`` is True, also returns a contrast-cut figure if diagnostics exist.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset from the bootstrap node.
    qubit_pairs : list
        Qubit pair objects for grid placement.
    fits : dict, optional
        Fit results keyed by pair name.
    analysis_debug : bool
        If True, include the contrast-cut diagnostic figure.
    cz_or_iswap : str
        ``"cz"`` or ``"iswap"`` — used in figure titles and gate-point legend.

    Returns
    -------
    dict[str, Figure]
        At least ``"target"`` and ``"control"``; optionally ``"contrast_debug"``.
    """
    pair_by_name = {qp.name: qp for qp in qubit_pairs}
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, Figure] = {}

    for state_type in ("target", "control"):
        grid = QubitPairGrid(grid_names, pair_names)
        for ax, qubit in grid_iter(grid):
            qp_name = qubit["qubit"]
            fit_data = fits.get(qp_name) if fits else None
            plot_individual_data_with_fit(
                ax,
                ds,
                qubit,
                fit_data,
                data_var=state_type,
                qubit_pair_obj=pair_by_name.get(qp_name),
                cz_or_iswap=cz_or_iswap,
            )

        grid.fig.suptitle(f"{'CZ' if cz_or_iswap == 'cz' else 'iSWAP'} flux landscape ({state_type})")
        grid.fig.tight_layout()
        figures[state_type] = grid.fig

    if analysis_debug:
        figures["contrast_debug"] = plot_contrast_cut_debug(ds, qubit_pairs, fits, cz_or_iswap=cz_or_iswap)

    return figures


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: Optional[Dict] = None,
    data_var: str = "target",
    qubit_pair_obj=None,
    cz_or_iswap: str = "cz",
):
    """Plot one qubit-pair 2D heatmap; optional fit and machine-state markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    ds : xr.Dataset
        Processed dataset with readout and flux coordinates.
    qubit : dict
        Grid entry with key ``"qubit"`` giving the pair name.
    fit : dict, optional
        Fit result dict for this pair (from ``fit_results``).
    data_var : str
        ``"target"`` or ``"control"`` readout to display.
    qubit_pair_obj : optional
        QUAM pair object; used to draw the current ``decouple_offset`` for reference.
    cz_or_iswap : str
        ``"cz"`` or ``"iswap"`` — legend label for the fitted gate coupler flux.
    """
    qubit_pair = qubit["qubit"]

    if data_var == "control":
        if "state_control" in ds:
            data = ds.state_control.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_control.sel(qubit_pair=qubit_pair)
    else:
        if "state_target" in ds:
            data = ds.state_target.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_target.sel(qubit_pair=qubit_pair)

    data.assign_coords(
        {
            "qubit_flux_mV": 1e3 * data.qubit_flux_full,
            "coupler_flux_mV": 1e3 * data.coupler_flux_full,
        }
    ).plot(ax=ax, x="qubit_flux_mV", y="coupler_flux_mV", cmap="viridis")

    gate_label = "CZ coupler flux" if cz_or_iswap == "cz" else "iSWAP coupler flux"
    has_legend = False

    if qubit_pair_obj is not None:
        idle_mV = 1e3 * qubit_pair_obj.coupler.decouple_offset
        if np.isfinite(idle_mV):
            ax.axhline(
                idle_mV,
                color="blue",
                lw=1.0,
                ls="--",
                label="Current decoupling offset",
            )
            has_legend = True

    if fit:
        decouple_flux = fit.get("optimal_decouple_offset", np.nan)
        if np.isfinite(decouple_flux):
            ax.axhline(
                1e3 * decouple_flux,
                color="green",
                lw=1.5,
                ls="--",
                label="Decoupling offset",
            )
            has_legend = True

        cz_total = fit.get("optimal_cz_coupler_flux_total", np.nan)
        if not np.isfinite(cz_total):
            cz_rel = fit.get("optimal_cz_coupler_flux", np.nan)
            if np.isfinite(cz_rel) and np.isfinite(decouple_flux):
                cz_total = decouple_flux + cz_rel
        if np.isfinite(cz_total):
            ax.axhline(
                1e3 * cz_total,
                color="red",
                lw=1.5,
                ls=":",
                label=gate_label,
            )
            has_legend = True

        qubit_flux = fit.get("optimal_qubit_flux", np.nan)
        if np.isfinite(qubit_flux):
            ax.axvline(
                1e3 * qubit_flux,
                color="white",
                lw=1.5,
                ls="-.",
                label="CZ qubit flux" if cz_or_iswap == "cz" else "iSWAP qubit flux",
            )
            has_legend = True

    ax.set_title(f"{qubit_pair} ({data_var})")

    if has_legend:
        ax.legend(fontsize=7, loc="upper right")

    _add_detuning_axis(ax, ds, qubit_pair, data)


def plot_contrast_cut_debug(
    ds: xr.Dataset,
    qubit_pairs: list,
    fits: Optional[Dict] = None,
    cz_or_iswap: str = "cz",
) -> Figure:
    """Plot contrast cuts with FFT flat/oscillation shading and fit markers.

    Requires contrast diagnostic arrays in ``fit_results`` (stored by ``analyse_data``).

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset (unused if diagnostics are stored in ``fits``).
    qubit_pairs : list
        Qubit pair objects for grid layout.
    fits : dict, optional
        Per-pair fit dicts including ``contrast_*`` and mask fields.

    Returns
    -------
    matplotlib.figure.Figure
        Grid figure with one contrast panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        fit = fits[qp_name] if fits else None
        if fit is None or fit.get("contrast_raw") is None:
            ax.set_title(f"{qp_name} (no contrast data)")
            continue

        if fit.get("contrast_coupler_full") is not None:
            x_v = np.asarray(fit["contrast_coupler_full"]).ravel()
        elif "qubit_pair" in ds.coupler_flux_full.dims:
            x_v = ds.coupler_flux_full.sel(qubit_pair=qp_name).values.ravel().astype(float)
        else:
            x_v = np.asarray(fit["contrast_coupler_rel"]).ravel()
        x = 1e3 * x_v
        y = np.asarray(fit["contrast_raw"]).ravel()
        smoothed = np.asarray(fit["contrast_smoothed"]).ravel()
        osc_mask = np.asarray(fit["osc_mask"]).astype(bool)
        flat_mask = np.asarray(fit["flat_mask"]).astype(bool)
        ax.plot(x, y, color="steelblue", lw=1.0, alpha=0.4, label="raw")
        ax.plot(x, smoothed, color="steelblue", lw=1.8, label="smoothed")
        ax.fill_between(
            x,
            0,
            1,
            where=osc_mask,
            alpha=0.12,
            color="tomato",
            transform=ax.get_xaxis_transform(),
            label="oscillation",
        )
        ax.fill_between(
            x,
            0,
            1,
            where=flat_mask,
            alpha=0.15,
            color="limegreen",
            transform=ax.get_xaxis_transform(),
            label="flat",
        )
        ax.axhline(0, color="gray", ls=":", lw=0.8)

        dec = fit.get("optimal_decouple_offset", np.nan)
        if np.isfinite(dec):
            ax.axvline(1e3 * dec, color="green", ls="--", lw=1.5, label="Decoupling offset")
        cz = fit.get("optimal_cz_coupler_flux_total", np.nan)
        if not np.isfinite(cz):
            cz_rel = fit.get("optimal_cz_coupler_flux", np.nan)
            if np.isfinite(cz_rel) and np.isfinite(dec):
                cz = dec + cz_rel
        if np.isfinite(cz):
            ax.axvline(1e3 * cz, color="red", ls="--", lw=1.5, label="CZ coupler flux" if cz_or_iswap == "cz" else "iSWAP coupler flux")

        qubit_v = fit.get("optimal_qubit_flux", np.nan)
        ax.set_xlabel("Coupler flux [mV]")
        ylab = "contrast (T−C)" if cz_or_iswap == "cz" else "contrast (C−T)"
        ax.set_ylabel(ylab)
        ax.legend(fontsize=6, loc="upper left")
        if np.isfinite(qubit_v):
            ax.set_title(f"{'CZ' if cz_or_iswap == 'cz' else 'iSWAP'} contrast cut @ qubit flux {float(qubit_v)*1e3:.1f} mV")
        else:
            ax.set_title(qp_name)

    gate = "CZ" if cz_or_iswap == "cz" else "iSWAP"
    grid.fig.tight_layout()
    return grid.fig


def _add_detuning_axis(ax: Axes, ds: xr.Dataset, qubit_pair: str, data: xr.DataArray) -> None:
    """Add a top secondary x-axis mapping qubit flux [mV] to detuning [MHz]."""
    if "detuning" not in ds:
        return
    try:
        detuning_data = ds.sel(qubit_pair=qubit_pair).detuning.values * 1e-6
        flux_qubit_data = (1e3 * data.qubit_flux_full.values).ravel()
        detuning_data = detuning_data.ravel()

        order = np.argsort(flux_qubit_data)
        x_sorted = flux_qubit_data[order]
        y_sorted = detuning_data[order]
        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]

        if x_unique.size >= 2:

            def flux_to_detuning(x):
                return np.interp(x, x_unique, y_unique)

            def detuning_to_flux(y):
                return np.interp(y, y_unique, x_unique)

            sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
            sec_ax.set_xlabel("Detuning [MHz]")
    except (ValueError, TypeError, IndexError):
        pass

    ax.set_xlabel("Qubit flux [mV]")
    ax.set_ylabel("Coupler flux [mV]")
