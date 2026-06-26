"""Figures for PSB readout-length sweeps (fidelity, visibility, summary, histograms)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from matplotlib.figure import Figure

from qualibrate.core import QualibrationNode

from calibration_utils.iq_sweep import (
    plot_fidelity_vs_sweep,
    plot_histograms_vs_sweep,
    plot_sweep_summary,
    plot_visibility_vs_sweep,
)

__all__ = ["plot_measure_duration_sweep_figures"]


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
        "fidelity_vs_sweep": plot_fidelity_vs_sweep(
            ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name
        ),
        "visibility_vs_sweep": plot_visibility_vs_sweep(
            ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name
        ),
        "sweep_summary": plot_sweep_summary(ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name),
        "histograms_vs_sweep": plot_histograms_vs_sweep(
            ds_raw, qubit_pairs, ds_fit, sweep_name=sweep_name, normalize_by_sweep=True
        ),
    }
