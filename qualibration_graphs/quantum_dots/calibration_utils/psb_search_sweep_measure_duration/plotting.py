"""Figures for PSB readout-length sweeps (fidelity, visibility, summary, histograms)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from matplotlib.figure import Figure
import xarray as xr

from qualibrate.core import QualibrationNode

from calibration_utils.common_utils.iq_utils import (
    plot_fidelity_vs_sweep,
    plot_histograms_vs_sweep,
    plot_sweep_summary,
    plot_visibility_vs_sweep,
    plot_rotated_iq_density_at_optimum,
)

__all__ = ["plot_measure_duration_sweep_figures", "plot_all"]


def plot_measure_duration_sweep_figures(
    node: QualibrationNode,
    *,
    sweep_name: Optional[str] = None,
) -> Dict[str, Figure]:
    """Build the same sweep figure set as detuning PSB (06a), vs readout length.

    Parameters
    ----------
    node
        Must have ``namespace['qubit_pairs']``, ``results['ds_raw']``, ``results['ds_fit']``.
    sweep_name
        Defaults to ``node.parameters.sweep_name``.
    """
    sweep_name = sweep_name or node.parameters.sweep_name
    qubit_pairs: List[Any] = node.namespace["qubit_pairs"]
    ds_raw = node.results["ds_raw"]
    ds_fit = node.results["ds_fit"]

    return {
        "fidelity_vs_sweep": plot_fidelity_vs_sweep(ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name),
        "visibility_vs_sweep": plot_visibility_vs_sweep(ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name),
        "sweep_summary": plot_sweep_summary(ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name),
        "histograms_vs_sweep": plot_histograms_vs_sweep(
            ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name, normalize_by_sweep=True
        ),
    }


def plot_all(
    ds_fit: xr.Dataset,
    qubit_pairs: Sequence[Union[str, Any]],
    fit_results: Dict[str, Any],
    *,
    ds_raw: xr.Dataset | None = None,
    sweep_name: str,
    plot_kde: bool = True,
    s: float = 4,
    alpha: float = 0.15,
) -> Dict[str, Figure]:
    """Generate all node figures for the PSB sweep using a 09/10-style node-facing API."""
    ds_raw = ds_fit if ds_raw is None else ds_raw
    qubit_pairs_list: List[Any] = list(qubit_pairs)
    figs = {
        "fidelity_vs_readout_length": plot_fidelity_vs_sweep(
            ds_raw, qubit_pairs_list, ds_fit, sweep_name=sweep_name, fit_results=fit_results
        ),
        "visibility_vs_readout_length": plot_visibility_vs_sweep(
            ds_raw, qubit_pairs_list, ds_fit, sweep_name=sweep_name, fit_results=fit_results
        ),
        "sweep_summary": plot_sweep_summary(
            ds_raw, qubit_pairs_list, ds_fit, sweep_name=sweep_name, fit_results=fit_results
        ),
        "histograms_vs_readout_length": plot_histograms_vs_sweep(
            ds_raw,
            qubit_pairs_list,
            ds_fit,
            sweep_name=sweep_name,
            normalize_by_sweep=True,
            fit_results=fit_results,
        ),
        "rotated_iq_density": plot_rotated_iq_density_at_optimum(
            ds_raw,
            fit_results,
            qubit_pairs_list,
            plot_kde=plot_kde,
            s=s,
            alpha=alpha,
        ),
    }
    return figs
