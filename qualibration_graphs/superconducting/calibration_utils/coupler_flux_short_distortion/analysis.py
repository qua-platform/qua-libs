"""Coupler cryoscope analysis: qubit front-end + dispersion-based flux inversion."""

from __future__ import annotations

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.flux_distortions import (
    frequency_to_coupler_flux,
    resolve_coupler_freq_flux_curve,
)
from calibration_utils.qubit_flux_short_distortion.analysis import (
    _extract_relevant_fit_parameters,
    cryoscope_frequency,
    fit_fir_data,
    log_fitted_results,
)
from qualibration_libs.analysis import fit_oscillation, unwrap_phase

__all__ = [
    "process_raw_dataset",
    "fit_raw_data",
    "fit_fir_data",
    "log_fitted_results",
]


def _warn_if_flux_clipped(flux_vals: np.ndarray, flux_bias: np.ndarray, label: str) -> int:
    """Report how many reconstructed flux samples were clamped to the dispersion-curve edges."""
    flux = np.asarray(flux_vals, dtype=float)
    lo, hi = float(np.nanmin(flux_bias)), float(np.nanmax(flux_bias))
    tol = 1e-12 + 1e-9 * max(abs(lo), abs(hi))
    n_pinned = int(np.sum((np.abs(flux - lo) <= tol) | (np.abs(flux - hi) <= tol)))
    if n_pinned:
        print(
            f"  WARNING [{label}]: {n_pinned}/{flux.size} flux samples clamped to the dispersion "
            f"range [{lo:.4f}, {hi:.4f}] V. Those points carry no information — the measured "
            f"frequency fell outside the loaded curve."
        )
    return n_pinned


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ to voltage for coupler cryoscope (pair-name ``qubit`` coordinate)."""
    if node.parameters.use_state_discrimination:
        return ds

    qubits = node.namespace["qubits"]
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in qubits],
        coords={"qubit": [str(v) for v in ds.qubit.values]},
        dims=["qubit"],
    )
    return ds.assign({key: ds[key] * 2**12 / readout_lengths for key in ("I", "Q") if key in ds})


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """Cryoscope phase → freq, then coupler dispersion curve → flux → multi-exp fit."""
    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise ValueError("Dataset must contain either 'I' or 'state' data")

    dafit = fit_oscillation(ds[data], "frame")
    daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")

    qubits = node.namespace["qubits"]
    qubit_pairs = node.namespace["qubit_pairs"]
    dim_names = [str(v) for v in ds[data].qubit.values]
    source = getattr(node.parameters, "freq_to_flux_source", "auto")

    ds_fit = cryoscope_frequency(daphi, sg_order=2, sg_range=3)
    ds_fit["flux_response"] = xr.full_like(ds_fit["freq"], np.nan, dtype=float)

    for i, dim_name in enumerate(dim_names):
        if i >= len(qubit_pairs):
            break
        q = qubits[i]
        coupler = qubit_pairs[i].coupler
        abs_freq_q = q.xy.RF_frequency - ds_fit["freq"].sel(qubit=dim_name).values * 1e9
        selected = resolve_coupler_freq_flux_curve(q, coupler, node, source, log_callable=node.log)
        curve = selected.curve
        if curve is not None:
            flux_bias, abs_peak = curve
            flux_vals = frequency_to_coupler_flux(abs_freq_q, (flux_bias, np.abs(abs_peak)))
            _warn_if_flux_clipped(flux_vals, flux_bias, dim_name)
            ds_fit["flux_response"].values[i, :] = flux_vals
        else:
            ds_fit["flux_response"].values[i, :] = np.nan

    return _extract_relevant_fit_parameters(ds_fit, node, dim_names=dim_names)
