"""Analysis for three-tone coupler spectroscopy with flux pulse (22a)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

LogCallable = Callable[[str], None]


@dataclass
class FitResults:
    """Per-pair coupler frequency extracted from the three-tone dip."""

    success: bool
    coupler_frequency_hz: float
    coupler_flux_v: float


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add derived coordinates and IQ amplitude when needed."""
    qubit_pairs = node.namespace["qubit_pairs"]
    dfs = node.namespace["dfs"]
    rf_start = node.parameters.rf_frequency_startpoint_in_hz
    rf_freq = np.array(
        [
            dfs + (rf_start if rf_start is not None else qp.coupler.RF_frequency)
            for qp in qubit_pairs
        ]
    )
    ds = ds.assign_coords(freq_full_control=(["qubit", "freq"], rf_freq))
    ds.freq_full_control.attrs["long_name"] = "Coupler drive frequency"
    ds.freq_full_control.attrs["units"] = "Hz"

    flux_values = np.array(
        [node.parameters.coupler_flux_in_v + qp.coupler.decouple_offset for qp in qubit_pairs],
        dtype=float,
    )
    ds = ds.assign_coords(flux_full=(["qubit"], flux_values))
    ds.flux_full.attrs["long_name"] = "Coupler flux bias"
    ds.flux_full.attrs["units"] = "V"

    if not node.parameters.use_state_discrimination and "IQ_abs" not in ds:
        ds = ds.assign(IQ_abs=np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2))
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, Dict[str, FitResults]]:
    """Locate the coupler resonance from the target-qubit response vs drive frequency."""
    if node.parameters.use_state_discrimination:
        min_idx = ds.state.argmin(dim="freq")
        signal_name = "state"
    else:
        min_idx = ds.I.argmax(dim="freq")
        signal_name = "I"

    fit_results: Dict[str, FitResults] = {}
    for qp in node.namespace["qubit_pairs"]:
        pair_name = qp.name
        freq_hz = float(ds.freq_full_control.sel(qubit=pair_name).isel(freq=min_idx.sel(qubit=pair_name)).values)
        flux_v = float(ds.flux_full.sel(qubit=pair_name).values)
        fit_results[pair_name] = FitResults(
            success=np.isfinite(freq_hz),
            coupler_frequency_hz=freq_hz,
            coupler_flux_v=flux_v,
        )

    ds_fit = ds.assign_attrs(signal_used=signal_name)
    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable: Optional[LogCallable] = None) -> None:
    """Log extracted coupler frequencies."""
    log = log_callable or print
    for pair_name, fit in fit_results.items():
        if fit.success:
            log(
                f"{pair_name}: coupler frequency = {fit.coupler_frequency_hz * 1e-9:.4f} GHz "
                f"at flux {fit.coupler_flux_v * 1e3:.2f} mV"
            )
        else:
            log(f"{pair_name}: coupler frequency extraction failed")
