from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import xarray as xr


def map_estimates_from_raw(
    ds: xr.Dataset,
    qubit_names: List[str],
    v_f: np.ndarray,
    *,
    dim_repetition: str = "repetition",
    dim_tau: str = "tau",
    dim_frequency: str = "frequency",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Compute MAP frequency hypothesis per qubit from posterior streams Pf*.

    Averages over repetitions, takes the posterior slice at the last ``tau``,
    then argmax along the frequency axis.
    """
    fit_results: Dict[str, Dict[str, Any]] = {}
    estimates: Dict[str, float] = {}

    for qi, qname in enumerate(qubit_names, start=1):
        pf_name = f"Pf{qi}"
        if pf_name not in ds.data_vars:
            fit_results[qname] = {"success": False, "error": f"missing {pf_name}"}
            continue

        da = ds[pf_name]
        try:
            if dim_repetition in da.dims:
                collapsed = da.mean(dim=dim_repetition)
            else:
                collapsed = da

            if "qubit" in collapsed.dims:
                collapsed = collapsed.sel(qubit=qname, drop=True)

            if dim_tau in collapsed.dims:
                collapsed = collapsed.isel({dim_tau: -1})

            if dim_frequency not in collapsed.dims:
                fit_results[qname] = {
                    "success": False,
                    "error": f"missing dim {dim_frequency!r}, have {collapsed.dims!r}",
                }
                continue

            idx = int(np.nanargmax(collapsed.values))
            f_map = float(v_f[idx])
            estimates[qname] = f_map
            fit_results[qname] = {
                "success": True,
                "map_frequency": f_map,
                "map_index": idx,
            }
        except (ValueError, IndexError, KeyError) as exc:
            fit_results[qname] = {"success": False, "error": str(exc)}

    return fit_results, estimates
