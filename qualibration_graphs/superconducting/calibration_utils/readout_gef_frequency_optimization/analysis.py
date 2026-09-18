"""Analysis utilities for GEF readout frequency optimization: centroid distance fitting."""

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
    """Stores the relevant readout frequency optimization fit parameters for a single qubit."""

    optimal_detuning: float
    success: bool
    prominence_snr: float = float("nan")
    note: str = ""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted GEF frequency-shift results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tOptimal frequency shift: {1e-6 * fit_results[q]['optimal_detuning']:.3f} MHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        note = fit_results[q].get("note") or ""
        log_callable(s_qubit + s_freq + (f"\t{note}" if note else ""))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw IQ data to voltage for all g, e, f states."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Find the optimal GEF readout detuning (argmax of min g/e/f centroid distance)."""
    ds = ds.groupby("qubit").apply(fit_routine, node=node)
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for q in node.namespace["qubits"]:
        q = q.name
        if q not in ds_fit.qubit.values:
            logging.warning(f"Qubit {q} not found in the fit results.")
            continue

        def _get(name, default):
            try:
                return type(default)(ds_fit[name].sel(qubit=q).item())
            except Exception:
                return default

        fit_results[q] = FitParameters(
            optimal_detuning=ds_fit.optimal_detuning.sel(qubit=q).item(),
            success=bool(ds_fit.success.sel(qubit=q).item()),
            prominence_snr=_get("prominence_snr", float("nan")),
            note=_get("opt_note", ""),
        )
    return ds_fit, fit_results


def fit_routine(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Grade the argmax of the smoothed min pairwise g/e/f centroid distance."""
    del node
    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.Ig - ds.Ie) ** 2 + (ds.Qg - ds.Qe) ** 2),
            "Def": np.sqrt((ds.Ie - ds.If) ** 2 + (ds.Qe - ds.Qf) ** 2),
            "Dgf": np.sqrt((ds.Ig - ds.If) ** 2 + (ds.Qg - ds.Qf) ** 2),
        }
    )
    ds["Distance"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")
    smoothed = ds.Distance.rolling({"frequency": 3}, center=True, min_periods=1).mean("frequency")
    ds["Distance_smooth"] = smoothed
    detuning = smoothed.idxmax("frequency")
    quality = argmax_with_quality(ds.frequency.values, np.asarray(smoothed.data, dtype=float).reshape(-1))
    if quality.note:
        logging.getLogger(__name__).warning("gef freq opt %s: %s", str(np.atleast_1d(ds.qubit.data)[0]), quality.note)
    return ds.assign(
        {
            "optimal_detuning": detuning,
            "success": bool(quality.success),
            "prominence_snr": quality.prominence_snr,
            "opt_note": quality.note,
        }
    )
