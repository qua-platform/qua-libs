from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import xarray as xr


@dataclass
class FitParameters:
    optimal_ramp_duration: int
    optimal_wait_duration: int
    optimal_avg_state: float
    find_minimum: bool
    success: bool
    failure_reason: Optional[str] = None


def analyse_init_2d(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    find_minimum: bool = True,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Analyse a (ramp_duration × wait_duration) 2D state map per qubit pair.

    Failure conditions (success=False):
    - missing required variables/coords (`state_{qp}`, `ramp_duration`, `wait_duration`)
    - empty sweep axes
    - all-NaN / non-finite state map
    """
    if "ramp_duration" not in ds_raw.coords and "ramp_duration" not in ds_raw:
        raise KeyError("Expected `ramp_duration` coordinate in dataset.")
    if "wait_duration" not in ds_raw.coords and "wait_duration" not in ds_raw:
        raise KeyError("Expected `wait_duration` coordinate in dataset.")

    ramp = np.asarray(ds_raw["ramp_duration"].values)
    wait = np.asarray(ds_raw["wait_duration"].values)
    if ramp.size < 1 or wait.size < 1:
        raise ValueError("Empty sweep axis: ramp_duration or wait_duration.")

    fit_results: Dict[str, FitParameters] = {}

    opt_ramp_list: list[int] = []
    opt_wait_list: list[int] = []
    opt_state_list: list[float] = []
    success_list: list[bool] = []

    for qp in qubit_pair_names:
        key = f"state_{qp}"
        if key not in ds_raw:
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_wait_duration=int(wait[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason=f"Missing dataset variable `{key}`.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_wait_list.append(int(wait[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        state_2d = np.asarray(ds_raw[key].values, dtype=float)
        if state_2d.size == 0:
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_wait_duration=int(wait[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason="Empty state array.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_wait_list.append(int(wait[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        finite = np.isfinite(state_2d)
        if not np.any(finite):
            fit_results[qp] = FitParameters(
                optimal_ramp_duration=int(ramp[0]),
                optimal_wait_duration=int(wait[0]),
                optimal_avg_state=float("nan"),
                find_minimum=find_minimum,
                success=False,
                failure_reason="State map contains no finite values.",
            )
            opt_ramp_list.append(int(ramp[0]))
            opt_wait_list.append(int(wait[0]))
            opt_state_list.append(float("nan"))
            success_list.append(False)
            continue

        # nanargmin/nanargmax will ignore NaNs (but we already checked not all-NaN).
        opt_flat = int(np.nanargmin(state_2d)) if find_minimum else int(np.nanargmax(state_2d))
        opt_ramp_idx, opt_wait_idx = np.unravel_index(opt_flat, state_2d.shape)

        opt_ramp = int(ramp[opt_ramp_idx])
        opt_wait = int(wait[opt_wait_idx])
        opt_state = float(state_2d[opt_ramp_idx, opt_wait_idx])

        fit_results[qp] = FitParameters(
            optimal_ramp_duration=opt_ramp,
            optimal_wait_duration=opt_wait,
            optimal_avg_state=opt_state,
            find_minimum=find_minimum,
            success=True,
        )
        opt_ramp_list.append(opt_ramp)
        opt_wait_list.append(opt_wait)
        opt_state_list.append(opt_state)
        success_list.append(True)

    # Minimal ds_fit: keep ds_raw immutable, add per-qubit-pair summary vectors for plotting/UX.
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
            "optimal_wait_duration": xr.DataArray(
                opt_wait_list,
                dims=("qubit_pair",),
                coords={"qubit_pair": qp_coord},
                attrs={"long_name": "optimal wait duration", "units": "ns"},
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
                f"wait={r['optimal_wait_duration']} ns "
                f"({extremum} avg state = {r['optimal_avg_state']:.4f})"
            )
        else:
            reason = r.get("failure_reason", "analysis failed")
            log_callable(f"  {qp_name}: FAIL ({reason})")
