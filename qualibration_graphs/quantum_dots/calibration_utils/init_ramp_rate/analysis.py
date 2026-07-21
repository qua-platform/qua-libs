from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr


def analyse_ramp_rate(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    find_minimum: bool = True,
) -> Tuple[xr.Dataset, Dict]:
    """Identify the ramp duration that minimises (or maximises) the average state assignment.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``state_{pair_name}`` variables indexed by ``ramp_duration``.
    qubit_pair_names : list[str]
        Pair names to analyse.
    find_minimum : bool
        If *True* pick the ramp duration with the lowest average assignment
        (purest ground state); if *False* pick the highest.

    Returns
    -------
    ds_raw : xr.Dataset
        Unchanged input (kept for pipeline consistency).
    fit_results : dict
        Per-pair dict with ``optimal_ramp_duration``, ``optimal_avg_state``, and ``success``.
    """
    fit_results: Dict = {}

    for qp_name in qubit_pair_names:
        state = ds_raw[f"state_{qp_name}"]
        if "shot" in state.dims:
            avg_state = state.mean(dim="shot").values
        else:
            avg_state = state.values
        ramp_durations = ds_raw["ramp_duration"].values

        opt_idx = int(np.argmin(avg_state) if find_minimum else np.argmax(avg_state))

        fit_results[qp_name] = {
            "success": True,
            "optimal_ramp_duration": int(ramp_durations[opt_idx]),
            "optimal_avg_state": float(avg_state[opt_idx]),
            "find_minimum": find_minimum,
        }

    return ds_raw, fit_results


def log_fitted_results(fit_results: Dict, log_callable=print) -> None:
    """Log a human-readable summary of the ramp-rate analysis."""
    for qp_name, result in fit_results.items():
        if result["success"]:
            extremum = "minimum" if result["find_minimum"] else "maximum"
            log_callable(
                f"  {qp_name}: optimal ramp duration = {result['optimal_ramp_duration']} ns "
                f"({extremum} avg state = {result['optimal_avg_state']:.4f})"
            )
        else:
            log_callable(f"  {qp_name}: analysis failed")
