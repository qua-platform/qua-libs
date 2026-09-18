"""Analysis utilities for GEF readout power optimization."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from calibration_utils.common_utils.curve_quality import argmax_with_quality
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fitted GEF readout power optimization parameters for a single qubit."""

    optimal_amp_prefactor: float
    optimal_amplitude: float
    success: bool
    prominence_snr: float = float("nan")
    note: str = ""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log fitted readout power results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_amp = (
            f"\tOptimal readout amplitude prefactor: {fit_results[q]['optimal_amp_prefactor']:.4f} | "
            f"absolute amplitude: {1e3 * fit_results[q]['optimal_amplitude']:.3f} mV"
        )
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        note = fit_results[q].get("note") or ""
        log_callable(s_qubit + s_amp + (f"\n\t{note}" if note else ""))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw IQ data to volts and add absolute readout amplitude."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    readout_amplitudes = np.array(
        [
            ds.amp_prefactor * q.resonator.operations[node.parameters.operation].amplitude
            for q in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit centroid-distance data and extract optimal power for each qubit."""
    ds_fit = ds.groupby("qubit").apply(fit_routine, node=node)
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset):
    """Extract fit parameters into a typed dictionary."""
    fit_results = {}
    for q in ds_fit.qubit.values:

        def _get(name, default):
            try:
                return type(default)(ds_fit[name].sel(qubit=q).item())
            except Exception:
                return default

        fit_results[q] = FitParameters(
            optimal_amp_prefactor=float(ds_fit.optimal_amp_prefactor.sel(qubit=q).item()),
            optimal_amplitude=float(ds_fit.optimal_amplitude.sel(qubit=q).item()),
            success=bool(ds_fit.success.sel(qubit=q).item()),
            prominence_snr=_get("prominence_snr", float("nan")),
            note=_get("opt_note", ""),
        )
    return ds_fit, fit_results


def fit_routine(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Argmax of the smoothed min pairwise-separation curve, gated by prominence."""
    del node
    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.Ig - ds.Ie) ** 2 + (ds.Qg - ds.Qe) ** 2),
            "Def": np.sqrt((ds.Ie - ds.If) ** 2 + (ds.Qe - ds.Qf) ** 2),
            "Dgf": np.sqrt((ds.Ig - ds.If) ** 2 + (ds.Qg - ds.Qf) ** 2),
        }
    )
    ds["Distance"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")
    smoothed_distance = ds.Distance.rolling({"amp_prefactor": 3}, center=True, min_periods=1).mean()
    ds["Distance_smooth"] = smoothed_distance
    quality = argmax_with_quality(
        ds.amp_prefactor.values, np.asarray(smoothed_distance.data, dtype=float).reshape(-1)
    )
    optimal_amp_prefactor = smoothed_distance.idxmax("amp_prefactor")
    optimal_amplitude = ds.readout_amplitude.sel(amp_prefactor=optimal_amp_prefactor)
    success = bool(quality.success and np.all(np.isfinite(np.atleast_1d(optimal_amplitude.data))))
    if quality.note:
        logging.getLogger(__name__).warning(
            "gef power opt %s: %s", str(np.atleast_1d(ds.qubit.data)[0]), quality.note
        )
    return ds.assign(
        {
            "optimal_amp_prefactor": optimal_amp_prefactor,
            "optimal_amplitude": optimal_amplitude,
            "success": success,
            "prominence_snr": quality.prominence_snr,
            "opt_note": quality.note,
        }
    )
