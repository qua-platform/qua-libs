"""Plotting utilities for coupler flux long distortion (qubitspec variant)."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from calibration_utils.qubit_flux_long_distortion_qubitspec.plotting import (
    plot_center_freq,
    plot_flux_response,
    plot_freq_vs_flux_curve,
    plot_iq_abs,
)
from qualibration_libs.plotting import grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs,
    measured_qubits,
    fit_results: Dict,
    *,
    debug: bool = False,
    log_scale: bool = False,
) -> Dict[str, plt.Figure]:
    """Default figures: flux response (with IIR fit) only.

    With ``debug=True``: spectroscopy heatmap (or center frequency trace),
    and the coupler dispersion curve used for freq→flux inversion when available.

    Subplots are laid out with :class:`~calibration_utils.pair_grid.QubitPairGrid`.
    The dataset ``qubit`` coordinate is the pair name.
    """
    rf_by_name = {
        qp.name: getattr(getattr(mq, "xy", None), "RF_frequency", None)
        for qp, mq in zip(qubit_pairs, measured_qubits)
    }

    g_names, qp_names = grid_pair_names(qubit_pairs)
    figures: Dict[str, plt.Figure] = {}

    grid_flux = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid_flux):
        plot_flux_response(
            ax,
            ds,
            qubit,
            fit=fit_results.get(qubit["qubit"]),
            log_scale=log_scale,
        )
    grid_flux.fig.suptitle("Flux response vs time after flux pulse", fontsize=16)
    grid_flux.fig.tight_layout()
    figures["flux_response"] = grid_flux.fig

    if not debug:
        return figures

    has_spectroscopy_map = "IQ_abs" in ds.data_vars or "state" in ds.data_vars
    if has_spectroscopy_map and "center_freqs" in ds:
        title_label = "|IQ|" if "IQ_abs" in ds.data_vars else "State"
        grid_spec = QubitPairGrid(g_names, qp_names)
        for ax, qubit in grid_iter(grid_spec):
            plot_iq_abs(
                ax,
                ds,
                qubit,
                grid_spec.fig,
                log_scale=log_scale,
                overlay_center_freq=True,
                rf_frequency=rf_by_name.get(qubit["qubit"]),
            )
        grid_spec.fig.suptitle(
            f"Debug: {title_label} vs (time, frequency) with extracted center frequency",
            fontsize=16,
        )
        grid_spec.fig.tight_layout()
        figures["spectroscopy"] = grid_spec.fig
    elif "center_freqs" in ds:
        grid_cf = QubitPairGrid(g_names, qp_names)
        for ax, qubit in grid_iter(grid_cf):
            plot_center_freq(
                ax,
                ds,
                qubit,
                rf_frequency=rf_by_name.get(qubit["qubit"]),
                log_scale=log_scale,
            )
        grid_cf.fig.suptitle("Debug: Qubit frequency shift vs time after flux pulse", fontsize=16)
        grid_cf.fig.tight_layout()
        figures["center_freq"] = grid_cf.fig

    if "spec_curve_flux" in ds and "spec_curve_freq" in ds:
        grid_curve = QubitPairGrid(g_names, qp_names)
        source_label = ds.attrs.get("freq_to_flux_sources", ds.attrs.get("freq_to_flux_source", "measured"))
        spec_qubits = ds["spec_curve_flux"].spec_qubit.values.tolist()
        n_plotted = 0
        for ax, qubit in grid_iter(grid_curve):
            qname = qubit["qubit"]
            plot_freq_vs_flux_curve(ax, ds, qubit, source_label=source_label)
            if qname in spec_qubits:
                flux_arr = ds["spec_curve_flux"].sel(spec_qubit=qname).values
                freq_arr = ds["spec_curve_freq"].sel(spec_qubit=qname).values
                if np.isfinite(flux_arr).any() and np.isfinite(freq_arr).any():
                    n_plotted += 1
        if n_plotted:
            grid_curve.fig.suptitle(f"Debug: Coupler dispersion curve used ({source_label})", fontsize=16)
            grid_curve.fig.tight_layout()
            figures["freq_vs_flux_curve"] = grid_curve.fig
        else:
            plt.close(grid_curve.fig)

    return figures
