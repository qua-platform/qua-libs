"""Analysis for coupler flux long distortion (Ramsey variant).

Same pipeline as 17b (Ramsey phase → amplitude via reference sweep), but the long
pulse is on the coupler line and the dataset ``qubit`` coord is keyed by pair name.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from calibration_utils.common_utils.flux_distortions import FitParameters, multi_exp_fit_global
from calibration_utils.qubit_flux_long_distortion_ramsey.analysis import (
    _compute_flux_response,
    extract_phases,
)

# Re-export for callers that import from this package.
__all__ = ["process_raw_dataset", "fit_raw_data", "extract_phases"]


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V (pair-name ``qubit`` coord)."""
    if "I" in ds or "Q" in ds:
        qubit_pairs = node.namespace["qubit_pairs"]
        measured_qubits = node.namespace["qubits"]
        readout_lengths = xr.DataArray(
            [q.resonator.operations["readout"].length for q in measured_qubits],
            coords=[("qubit", [qp.name for qp in qubit_pairs])],
        )
        for key in ("I", "Q"):
            if key in ds:
                ds = ds.assign({key: ds[key] * 2**12 / readout_lengths})
    return ds


def _extract_relevant_fit_parameters(
    ds: xr.Dataset,
    node,
    coord_names: list[str],
) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit coupler flux step response per qubit pair (``qubit`` coord = pair name)."""
    n_exponentials = int(node.parameters.n_exponentials)
    t_pulse_ns = float(getattr(node.parameters, "flux_settle_time_in_ns", 0)) or None
    flux_response = ds["flux_response"]
    fit_results: Dict[str, FitParameters] = {}
    min_points = max(2 * n_exponentials + 1, 4)

    for name in coord_names:
        qf = flux_response.sel(qubit=name)
        t_data = np.asarray(qf.time.values, dtype=float)
        y_data = np.asarray(qf.values, dtype=float)
        mask = np.isfinite(y_data) & (t_data > 0)
        if mask.sum() < min_points:
            fit_results[name] = FitParameters(
                success=False,
                n_components_requested=n_exponentials,
                n_components_used=0,
                a_tau_tuple=[],
                a_dc=float("nan"),
                rms_error=float("nan"),
            )
            continue
        fit_results[name] = multi_exp_fit_global(
            t_data[mask],
            y_data[mask],
            n_exponentials=n_exponentials,
            t_pulse_ns=t_pulse_ns,
            verbose=True,
            log_callable=node.log,
        )
    return ds, fit_results


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract phases, map to coupler flux response, and fit exponential cascade."""
    qubit_pairs = node.namespace["qubit_pairs"]
    coord_names = [qp.name for qp in qubit_pairs]

    signal_phase, ref_cal = extract_phases(ds)
    flux_response, branch_risk = _compute_flux_response(
        signal_phase,
        ref_cal,
        qubit_pairs,
        ramsey_flux_amp=node.parameters.ramsey_flux_amplitude_in_v,
        qubit_flux_amp=getattr(node.parameters, "coupler_flux_amplitude_in_v", None),
        log_callable=node.log,
    )

    ds = ds.copy()
    ds["signal_phase"] = signal_phase
    if ref_cal is not None:
        ds["ref_phase_cal"] = ref_cal
    ds["flux_response"] = flux_response
    if ref_cal is not None and all(n in branch_risk for n in coord_names):
        ds["branch_risk_code"] = xr.DataArray(
            [branch_risk[n]["code"] for n in coord_names],
            dims=["qubit"],
            coords={"qubit": coord_names},
        )
        ds["branch_sig_swing"] = xr.DataArray(
            [branch_risk[n]["sig_swing_frac"] for n in coord_names],
            dims=["qubit"],
            coords={"qubit": coord_names},
            attrs={"long_name": "signal phase peak-to-peak swing", "units": "2*pi"},
        )
        ds["branch_ref_span"] = xr.DataArray(
            [branch_risk[n]["ref_span_frac"] for n in coord_names],
            dims=["qubit"],
            coords={"qubit": coord_names},
            attrs={"long_name": "reference phase span", "units": "2*pi"},
        )
        ds["branch_out_of_range"] = xr.DataArray(
            [branch_risk[n]["oor_frac"] for n in coord_names],
            dims=["qubit"],
            coords={"qubit": coord_names},
            attrs={"long_name": "delay points outside the reference phase window", "units": "fraction"},
        )

    return _extract_relevant_fit_parameters(ds, node, coord_names=coord_names)
