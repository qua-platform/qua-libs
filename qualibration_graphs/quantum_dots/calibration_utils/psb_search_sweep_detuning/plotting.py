"""Plotting API for PSB detuning sweep (06a).

The underlying plots are provided by the shared `iq_sweep` plotting utilities.
This module exposes a single `plot_all` entrypoint so nodes depend only on their
local calibration_utils package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.iq_utils.iq_sweep.plotting import (
    plot_fidelity_vs_sweep,
    plot_histograms_vs_sweep,
    plot_sweep_summary,
    plot_rotated_iq_density_at_optimum,
    plot_visibility_vs_sweep,
)


def plot_all(
    ds_raw: xr.Dataset,
    qubit_pairs: Sequence[Union[str, Any]],
    ds_fit: xr.Dataset,
    *,
    sweep_name: str = "detuning",
    fit_results: Optional[Dict[str, Any]] = None,
    plot_kde: bool = True,
    s: float = 4,
    alpha: float = 0.15,
) -> Dict[str, Figure]:
    """Generate all standard figures for the PSB detuning sweep node."""
    if fit_results is None:
        raise ValueError("fit_results must be provided for rotated IQ density plot.")

    figures: Dict[str, Figure] = {}
    figures["fidelity_vs_detuning"] = plot_fidelity_vs_sweep(
        ds_raw,
        list(qubit_pairs),
        ds_fit,
        sweep_name=sweep_name,
    )
    figures["visibility_vs_detuning"] = plot_visibility_vs_sweep(
        ds_raw,
        list(qubit_pairs),
        ds_fit,
        sweep_name=sweep_name,
    )
    figures["sweep_summary"] = plot_sweep_summary(
        ds_raw,
        list(qubit_pairs),
        ds_fit,
        sweep_name=sweep_name,
    )
    figures["histograms_vs_detuning"] = plot_histograms_vs_sweep(
        ds_raw,
        list(qubit_pairs),
        ds_fit,
        sweep_name=sweep_name,
    )
    figures["rotated_iq_density"] = plot_rotated_iq_density_at_optimum(
        ds_raw,
        fit_results,
        list(qubit_pairs),
        plot_kde=plot_kde,
        s=s,
        alpha=alpha,
    )
    return figures


__all__ = ["plot_all"]
