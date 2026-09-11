"""Analysis for three-tone coupler spectroscopy vs coupler flux (22b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from calibration_utils.three_tone_coupler_spectroscopy_flux_pulse.parameters import (
    resolve_coupler_rf_centers_by_pair,
)
from qualibrate import QualibrationNode

LogCallable = Callable[[str], None]


@dataclass
class FitResults:
    """Per-pair tracking resonance vs flux (optional summary)."""

    success: bool
    coupler_frequency_hz: float
    coupler_flux_v: float


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add absolute frequency coordinates and IQ amplitude when needed."""
    qubit_pairs = node.namespace["qubit_pairs"]
    dfs = node.namespace["dfs"]
    coupler_rf_centers = node.namespace.get("coupler_rf_centers") or resolve_coupler_rf_centers_by_pair(
        qubit_pairs,
        node.parameters.rf_frequency_startpoint_in_hz,
        coupler_band=node.parameters.coupler_band,
    )
    rf_freq = np.array([dfs + coupler_rf_centers[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords(freq_full_control=(["qubit", "freq"], rf_freq))
    ds.freq_full_control.attrs["long_name"] = "Coupler drive frequency"
    ds.freq_full_control.attrs["units"] = "Hz"

    if not node.parameters.use_state_discrimination and "IQ_abs" not in ds:
        ds = ds.assign(IQ_abs=np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2))
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, Dict[str, FitResults]]:
    """Track the minimum-response coupler frequency at each flux bias."""
    if node.parameters.use_state_discrimination:
        min_idx = ds.state.argmin(dim="freq")
    else:
        min_idx = ds.IQ_abs.argmin(dim="freq")

    min_freqs = ds.freq_full_control.isel(freq=min_idx)
    fit_results: Dict[str, FitResults] = {}
    flux_coord = "flux" if "flux" in ds.dims else "coupler_flux"

    for qp in node.namespace["qubit_pairs"]:
        pair_name = qp.name
        freq_trace = min_freqs.sel(qubit=pair_name)
        flux_vals = ds[flux_coord].values
        best_i = int(np.nanargmin(freq_trace.values)) if freq_trace.size else 0
        freq_hz = float(freq_trace.isel({flux_coord: best_i}).values)
        flux_v = float(flux_vals[best_i]) if len(flux_vals) else np.nan
        fit_results[pair_name] = FitResults(
            success=np.isfinite(freq_hz),
            coupler_frequency_hz=freq_hz,
            coupler_flux_v=flux_v,
        )

    return ds, fit_results


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable: Optional[LogCallable] = None) -> None:
    """Log a single reference point from the 2D map per pair."""
    log = log_callable or print
    for pair_name, fit in fit_results.items():
        if fit.success:
            log(
                f"{pair_name}: reference coupler frequency = {fit.coupler_frequency_hz * 1e-9:.4f} GHz "
                f"at flux {fit.coupler_flux_v * 1e3:.2f} mV"
            )
        else:
            log(f"{pair_name}: coupler frequency extraction failed")
