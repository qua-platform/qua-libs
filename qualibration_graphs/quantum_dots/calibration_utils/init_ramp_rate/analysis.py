from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import xarray as xr


@dataclass
class FitParameters:
    optimal_ramp_duration: int
    optimal_avg_state: float
    find_minimum: bool
    success: bool
    failure_reason: Optional[str] = None


def analyse_ramp_rate(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    find_minimum: bool = True,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
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
    ds_fit : xr.Dataset
        Copy of the raw dataset with per-qubit-pair summary vectors attached.
    fit_results : dict
        Per-pair FitParameters with ``optimal_ramp_duration``, ``optimal_avg_state``, and ``success``.
    """
    if "ramp_duration" not in ds_raw:
        raise KeyError("Expected `ramp_duration` coordinate in dataset.")

    ramp_durations = np.asarray(ds_raw["ramp_duration"].values)
    if ramp_durations.size < 1:
        raise ValueError("Empty sweep axis: ramp_duration.")

    fit_results: Dict[str, FitParameters] = {}
    opt_ramp_list: list[int] = []
    opt_state_list: list[float] = []
    success_list: list[bool] = []

    for qp_name in qubit_pair_names:
        key = f"state_{qp_name}"
        if key not in ds_raw:
            fit_results[qp_name] = FitParameters(
                optimal_ramp_duration=int(ramp_durations[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason=f"Missing dataset variable `{key}`.",
            )
            opt_ramp_list.append(int(ramp_durations[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        state = ds_raw[key]
        if "shot" in state.dims:
            avg_state = state.mean(dim="shot").values
        else:
            avg_state = state.values
        avg_state = np.asarray(avg_state, dtype=float)

        finite = np.isfinite(avg_state)
        if not np.any(finite):
            fit_results[qp_name] = FitParameters(
                optimal_ramp_duration=int(ramp_durations[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason="Average-state trace contains no finite values.",
            )
            opt_ramp_list.append(int(ramp_durations[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        opt_idx = (
            int(np.nanargmin(avg_state)) if find_minimum else int(np.nanargmax(avg_state))
        )

        opt_ramp = int(ramp_durations[opt_idx])
        opt_state = float(avg_state[opt_idx])
        fit_results[qp_name] = FitParameters(
            optimal_ramp_duration=opt_ramp,
            optimal_avg_state=opt_state,
            find_minimum=find_minimum,
            success=True,
        )
        opt_ramp_list.append(opt_ramp)
        opt_state_list.append(opt_state)
        success_list.append(True)

    qp_coord = xr.DataArray(qubit_pair_names, dims=("qubit_pair",), name="qubit_pair")
    ds_fit = ds_raw.copy()
    ds_fit = ds_fit.assign(
        {
            "optimal_ramp_duration": xr.DataArray(
                opt_ramp_list,
                dims=("qubit_pair",),
                coords={"qubit_pair": qp_coord},
                attrs={"long_name": "optimal ramp duration", "units": "ns"},
            ),
            "optimal_avg_state": xr.DataArray(
                opt_state_list,
                dims=("qubit_pair",),
                coords={"qubit_pair": qp_coord},
                attrs={"long_name": "optimal average state", "units": "arb."},
            ),
            "success": xr.DataArray(
                success_list,
                dims=("qubit_pair",),
                coords={"qubit_pair": qp_coord},
                attrs={"long_name": "analysis success"},
            ),
        }
    )

    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict[str, dict], log_callable=print) -> None:
    """Log a human-readable summary of the ramp-rate analysis (expects serialized dicts)."""
    for qp_name, r in fit_results.items():
        if r.get("success", False):
            extremum = "minimum" if r.get("find_minimum", True) else "maximum"
            log_callable(
                f"  {qp_name}: optimal ramp duration = {r['optimal_ramp_duration']} ns "
                f"({extremum} avg state = {r['optimal_avg_state']:.4f})"
            )
        else:
            reason = r.get("failure_reason", "analysis failed")
            log_callable(f"  {qp_name}: FAIL ({reason})")
