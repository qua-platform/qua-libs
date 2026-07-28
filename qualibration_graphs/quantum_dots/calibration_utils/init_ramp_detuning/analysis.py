from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import xarray as xr


@dataclass
class FitParameters:
    optimal_ramp_duration: int
    optimal_detuning: float
    optimal_avg_state: float
    find_minimum: bool
    success: bool
    failure_reason: Optional[str] = None


def analyse_init_ramp_detuning(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    find_minimum: bool = True,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Analyse a (ramp_duration × detuning) 2D state map per qubit pair.

    Failure conditions (success=False):
    - missing required variables/coords (`state_{qp}`, `ramp_duration`, `detuning`)
    - empty sweep axes
    - all-NaN / non-finite state map
    """
    if "ramp_duration" not in ds_raw.coords and "ramp_duration" not in ds_raw:
        raise KeyError("Expected `ramp_duration` coordinate in dataset.")
    if "detuning" not in ds_raw.coords and "detuning" not in ds_raw:
        raise KeyError("Expected `detuning` coordinate in dataset.")

    ramp = np.asarray(ds_raw["ramp_duration"].values)
    det = np.asarray(ds_raw["detuning"].values, dtype=float)
    if ramp.size < 1 or det.size < 1:
        raise ValueError("Empty sweep axis: ramp_duration or detuning.")

    fit_results: Dict[str, FitParameters] = {}

    opt_ramp_list: list[int] = []
    opt_det_list: list[float] = []
    opt_state_list: list[float] = []
    success_list: list[bool] = []

    for qp in qubit_pair_names:
        key = f"state_{qp}"
        if key not in ds_raw:
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_detuning=float(det[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason=f"Missing dataset variable `{key}`.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_det_list.append(float(det[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        state_2d = np.asarray(ds_raw[key].values, dtype=float)
        if state_2d.size == 0:
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_detuning=float(det[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason="Empty state array.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_det_list.append(float(det[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        finite = np.isfinite(state_2d)
        if not np.any(finite):
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_detuning=float(det[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason="State map contains no finite values.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_det_list.append(float(det[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        opt_flat = (
            int(np.nanargmin(state_2d))
            if find_minimum
            else int(np.nanargmax(state_2d))
        )
        opt_ramp_idx, opt_det_idx = np.unravel_index(opt_flat, state_2d.shape)

        opt_ramp = int(ramp[opt_ramp_idx])
        opt_detuning = float(det[opt_det_idx])
        opt_state = float(state_2d[opt_ramp_idx, opt_det_idx])

        fit_results[qp] = FitParameters(
            optimal_ramp_duration=opt_ramp,
            optimal_detuning=opt_detuning,
            optimal_avg_state=opt_state,
            find_minimum=find_minimum,
            success=True,
        )
        opt_ramp_list.append(opt_ramp)
        opt_det_list.append(opt_detuning)
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
            "optimal_detuning": xr.DataArray(
                opt_det_list,
                dims=("qubit_pair",),
                coords={"qubit_pair": qp_coord},
                attrs={"long_name": "optimal detuning", "units": "V"},
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
    """Log node-specific fitted results for all qubit pairs (expects serialized dicts)."""
    for qp_name, r in fit_results.items():
        if r.get("success", False):
            extremum = "minimum" if r.get("find_minimum", True) else "maximum"
            log_callable(
                f"  {qp_name}: optimal ramp={r['optimal_ramp_duration']} ns, "
                f"detuning={r['optimal_detuning']:.4f} V "
                f"({extremum} avg state = {r['optimal_avg_state']:.4f})"
            )
        else:
            reason = r.get("failure_reason", "analysis failed")
            log_callable(f"  {qp_name}: FAIL ({reason})")

