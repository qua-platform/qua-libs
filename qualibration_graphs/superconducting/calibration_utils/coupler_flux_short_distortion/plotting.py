"""Plotting utilities for coupler flux short distortion (cryoscope)."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from calibration_utils.qubit_flux_short_distortion.plotting import (
    _debug_suptitle,
    _plot_cryoscope_freq_ax,
    _plot_flux_response_ax,
    _plot_unwrapped_phase_ax,
    plot_fir_resampled,
    plot_freq_vs_flux_curve,
)


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs,
    fit_results: dict,
    *,
    debug: bool = False,
    fir_results: Optional[dict] = None,
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default: flux response vs time (IIR fit; FIR overlay when ``fir_results`` given).

    Subplots are laid out with :class:`~calibration_utils.pair_grid.QubitPairGrid`.
    The dataset ``qubit`` coordinate is the pair name.

    With ``debug=True``: cryoscope frequency, unwrapped phase, freq-vs-flux curve,
    and FIR resampling check.
    """
    g_names, qp_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, plt.Figure] = {}
    fir_by_pair = fir_results or {}

    grid_flux = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid_flux):
        qname = qubit["qubit"]
        _plot_flux_response_ax(
            ax,
            ds_fit,
            qubit,
            fit=fit_results.get(qname),
            fir_result=fir_by_pair.get(qname),
            log_scale=log_scale,
        )
    title = "Flux response vs time"
    if any(res.get("success") for res in fir_by_pair.values()):
        title += " (with FIR overlay)"
    grid_flux.fig.suptitle(title, fontsize=16)
    grid_flux.fig.tight_layout()
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    grid_freq = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid_freq):
        _plot_cryoscope_freq_ax(ax, ds_fit, qubit, fit=fit_results.get(qubit["qubit"]), log_scale=log_scale)
    grid_freq.fig.suptitle(_debug_suptitle("Cryoscope frequency vs time"), fontsize=16)
    grid_freq.fig.tight_layout()
    figures["cryoscope_freq"] = grid_freq.fig

    if "phase" in ds_fit:
        grid_phase = QubitPairGrid(g_names, qp_names)
        for ax, qubit in grid_iter(grid_phase):
            _plot_unwrapped_phase_ax(ax, ds_fit, qubit)
        grid_phase.fig.suptitle(_debug_suptitle("Unwrapped phase vs time"), fontsize=16)
        grid_phase.fig.tight_layout()
        figures["unwrapped_phase"] = grid_phase.fig

    if "spec_curve_flux" in ds_fit and "spec_curve_freq" in ds_fit:
        source_label = ds_fit.attrs.get("freq_to_flux_sources", ds_fit.attrs.get("freq_to_flux_source", "measured"))
        grid_curve = QubitPairGrid(g_names, qp_names)
        n_plotted = 0
        for ax, qubit in grid_iter(grid_curve):
            plot_freq_vs_flux_curve(ax, ds_fit, qubit, source_label=source_label)
            qname = qubit["qubit"]
            if qname in ds_fit["spec_curve_flux"].spec_qubit.values.tolist():
                flux_arr = ds_fit["spec_curve_flux"].sel(spec_qubit=qname).values
                freq_arr = ds_fit["spec_curve_freq"].sel(spec_qubit=qname).values
                if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                    n_plotted += 1
        if n_plotted:
            grid_curve.fig.suptitle(_debug_suptitle(f"Freq-vs-flux curve used ({source_label})"), fontsize=16)
            grid_curve.fig.tight_layout()
            figures["freq_vs_flux_curve"] = grid_curve.fig
        else:
            plt.close(grid_curve.fig)

    if fir_results:
        grid_fir_r = QubitPairGrid(g_names, qp_names)
        n_fir = 0
        for ax, qubit in grid_iter(grid_fir_r):
            qname = qubit["qubit"]
            res = fir_results.get(qname)
            if res is not None and res.get("success"):
                plot_fir_resampled(ax, res)
                ax.set_title(qname)
                n_fir += 1
            else:
                ax.set_title(f"{qname} — no FIR")
        if n_fir:
            grid_fir_r.fig.suptitle(_debug_suptitle("FIR resampled flux"), fontsize=16)
            grid_fir_r.fig.tight_layout()
            figures["fir_resampled"] = grid_fir_r.fig
        else:
            plt.close(grid_fir_r.fig)

    return figures
