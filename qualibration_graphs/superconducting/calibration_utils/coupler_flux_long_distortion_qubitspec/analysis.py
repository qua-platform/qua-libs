"""Analysis for coupler flux long distortion (qubitspec variant).

Same pipeline as 17a (π vs flux spectroscopy), but freq→flux uses coupler
dispersion curves (03c / 09b) and returns absolute coupler flux.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import xarray as xr

from qualibration_libs.data import add_amplitude_and_phase

from calibration_utils.common_utils.flux_distortions import (
    CouplerFreqFluxSource,
    FitParameters,
    frequency_to_coupler_flux,
    multi_exp_fit_global,
    resolve_coupler_freq_flux_curve,
)
from calibration_utils.qubit_flux_long_distortion_qubitspec.analysis import (
    _attach_spec_curve_vars,
    extract_center_freqs,
    fit_raw_data as _pi_flux_fit_raw_data,
)

LogCallable = Callable[[str], None]


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V (pair-name ``qubit`` coord), add amplitude/phase, attach axes."""
    measured_qubits = node.namespace["qubits"]
    if "I" in ds or "Q" in ds:
        qubit_pairs = node.namespace["qubit_pairs"]
        readout_lengths = xr.DataArray(
            [q.resonator.operations["readout"].length for q in measured_qubits],
            coords=[("qubit", [qp.name for qp in qubit_pairs])],
        )
        for key in ("I", "Q"):
            if key in ds:
                ds = ds.assign({key: ds[key] * 2**12 / readout_lengths})
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace and node.namespace["sweep_axes"].get("detuning") is not None
        else np.asarray(ds["detuning"].values, dtype=float)
    )
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency for q in measured_qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.full((len(measured_qubits), len(dfs)), np.nan, dtype=float),
            ),
        }
    )
    ds.freq_full.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _compute_coupler_flux_response(
    center_freqs: xr.DataArray,
    qubits: list,
    qubit_pairs: list,
    node,
    freq_to_flux_source: CouplerFreqFluxSource = "auto",
    *,
    log_callable: LogCallable,
) -> Tuple[xr.DataArray, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, str]]:
    """Map ``center_freqs(t)`` to absolute coupler flux via the dispersion curve."""
    coord_names = [qp.name for qp in qubit_pairs[: len(qubits)]]
    flux_response = xr.full_like(center_freqs, np.nan, dtype=float)
    measured_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sources: Dict[str, str] = {}

    for i, (q, qp) in enumerate(zip(qubits, qubit_pairs)):
        coord = coord_names[i]
        selected = resolve_coupler_freq_flux_curve(
            q, qp.coupler, node, freq_to_flux_source, log_callable=log_callable
        )
        sources[coord] = selected.label

        if not selected.is_measured or selected.curve is None or len(selected.curve[0]) < 2:
            continue

        flux_bias, abs_peak = selected.curve
        measured_curves[coord] = (flux_bias, abs_peak)
        abs_freq_q = center_freqs.sel(qubit=coord).values + q.xy.RF_frequency
        flux_response.values[i, :] = frequency_to_coupler_flux(abs_freq_q, (flux_bias, abs_peak))

    return flux_response, measured_curves, sources


def _extract_relevant_fit_parameters(
    ds: xr.Dataset,
    node,
    coord_names: list[str],
) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit coupler flux step response per qubit pair (``qubit`` coord = pair name)."""
    n_exponentials = int(node.parameters.n_exponentials)
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
            verbose=True,
            log_callable=node.log,
        )
    return ds, fit_results


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract center frequencies, map to coupler flux, and fit exponential cascade."""
    qubits = node.namespace["qubits"]
    qubit_pairs = node.namespace.get("qubit_pairs")
    if qubit_pairs is None:
        return _pi_flux_fit_raw_data(ds, node)

    if "detuning" not in ds.dims and "freq" in ds.dims:
        ds = ds.rename({"freq": "detuning"})

    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else ds["detuning"].values
    )
    center_freqs = extract_center_freqs(
        ds,
        dfs,
        use_state_discrimination=bool(
            node.parameters.use_state_discrimination and "state" in ds.data_vars
        ),
        log_callable=node.log,
    )

    freq_to_flux_source = getattr(node.parameters, "freq_to_flux_source", "auto")
    flux_response, measured_curves, sources = _compute_coupler_flux_response(
        center_freqs,
        qubits,
        qubit_pairs,
        node,
        freq_to_flux_source=freq_to_flux_source,
        log_callable=node.log,
    )

    for coord, label in sources.items():
        node.log(f"  {coord}: freq→flux via {label}")

    coord_names = [qp.name for qp in qubit_pairs[: len(qubits)]]
    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response
    ds = _attach_spec_curve_vars(ds, measured_curves)
    ds.attrs["freq_to_flux_source"] = str(freq_to_flux_source)
    if sources:
        ds.attrs["freq_to_flux_sources"] = "; ".join(f"{k}: {v}" for k, v in sources.items())

    return _extract_relevant_fit_parameters(ds, node, coord_names=coord_names)
