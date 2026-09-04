"""Plotting utilities for coupler flux long distortion (Ramsey variant)."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr
from qualibration_libs.plotting import QubitGrid, grid_iter

from calibration_utils.qubit_flux_long_distortion_qubitspec.plotting import plot_flux_response
from calibration_utils.qubit_flux_long_distortion_ramsey.plotting import (
    annotate_branch_risk,
    plot_ramsey_fringe,
    plot_ref_phase_cal,
    plot_signal_phase,
)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs,
    fit_results: Dict,
    *,
    debug: bool = False,
    ds_raw: Optional[xr.Dataset] = None,
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default figures: flux response (with IIR fit) only.

    With ``debug=True``: Ramsey fringe heatmap (time × frame), extracted Ramsey
    phase vs time, and the reference phase-vs-amplitude calibration curve.
    """
    grid_locations = [qp.grid_location for qp in qubit_pairs]
    figures: Dict[str, plt.Figure] = {}

    grid_flux = QubitGrid(ds, grid_locations)
    for ax, qubit in grid_iter(grid_flux):
        plot_flux_response(
            ax,
            ds,
            qubit,
            fit=fit_results.get(qubit["qubit"]),
            log_scale=log_scale,
        )
    grid_flux.fig.suptitle("Flux response vs time after flux pulse", fontsize=16)
    grid_flux.fig.set_size_inches(15, 9)
    grid_flux.fig.tight_layout()
    annotate_branch_risk(grid_flux.fig, ds)
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    if ds_raw is not None:
        signal_key = "state" if "state" in ds_raw.data_vars else "I"
        if signal_key in ds_raw.data_vars and "frame" in ds_raw[signal_key].dims:
            grid_fringe = QubitGrid(ds_raw, grid_locations)
            for ax, qubit in grid_iter(grid_fringe):
                plot_ramsey_fringe(ax, ds_raw, qubit, grid_fringe.fig, log_scale=log_scale)
            grid_fringe.fig.suptitle("Debug: Ramsey signal vs (time, frame rotation)", fontsize=16)
            grid_fringe.fig.set_size_inches(15, 9)
            grid_fringe.fig.tight_layout()
            figures["ramsey_fringe"] = grid_fringe.fig

    if "signal_phase" in ds:
        grid_phase = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_phase):
            plot_signal_phase(ax, ds, qubit, log_scale=log_scale)
        grid_phase.fig.suptitle("Debug: Ramsey phase vs time after flux pulse", fontsize=16)
        grid_phase.fig.set_size_inches(15, 9)
        grid_phase.fig.tight_layout()
        figures["signal_phase"] = grid_phase.fig

    if "ref_phase_cal" in ds:
        grid_ref = QubitGrid(ds, grid_locations)
        for ax, qubit in grid_iter(grid_ref):
            plot_ref_phase_cal(ax, ds, qubit)
        grid_ref.fig.suptitle("Debug: Reference Ramsey phase vs flux amplitude", fontsize=16)
        grid_ref.fig.set_size_inches(15, 9)
        grid_ref.fig.tight_layout()
        figures["ref_phase_cal"] = grid_ref.fig

    return figures
