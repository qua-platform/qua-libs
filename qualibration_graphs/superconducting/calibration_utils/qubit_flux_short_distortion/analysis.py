"""Analysis utilities for cryoscope experiment: flux line step response fitting.

Cryoscope-specific front end (phase unwrap → dφ/dt → freq → flux), then the same
shared multi-exp IIR fit used by the long-distortion nodes (``FitParameters``,
``multi_exp_fit_global``). Optional FIR stage is short-only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.flux_distortions import FitParameters, multi_exp_fit_global
from calibration_utils.common_utils.flux_distortions.curves import (
    FreqFluxSource,
    frequency_to_flux_deviation,
    resolve_freq_flux_curve,
)
from qualibration_libs.analysis import fit_oscillation, unwrap_phase
from qualibration_libs.data import convert_IQ_to_V
from scipy.signal import savgol_filter


# --- Node-facing helpers (same layout as long-distortion analysis) ---


def log_fitted_results(fit_results: Dict, log_callable=print) -> None:
    """Log fitted cryoscope results for each qubit.

    Expects dict-form results (after ``asdict``), matching long-distortion nodes.
    """
    for qb, res in fit_results.items():
        if res["success"]:
            n_req = res.get("n_components_requested", "?")
            n_used = res.get("n_components_used", len(res["a_tau_tuple"]))
            log_callable(
                f"{qb}: SUCCESS (n_req={n_req}, n_used={n_used}), "
                f"a_dc={res['a_dc']:.3e}, rms={res['rms_error']:.3e}, "
                f"comps={res['a_tau_tuple']}"
            )
        else:
            log_callable(f"{qb}: FAILED")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ data to voltage if state discrimination is not used."""
    if node.parameters.use_state_discrimination:
        return ds
    return convert_IQ_to_V(ds, node.namespace["qubits"])


# --- Cryoscope frequency extraction ---


def savgol(da, dim, range=3, order=2):
    """Apply Savitzky-Golay filter to smooth data."""

    def diff_func(x):
        return savgol_filter(x, range, order, deriv=0, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def diff_savgol(da, dim, range=3, order=2):
    """Apply Savitzky-Golay filter to compute derivative."""

    def diff_func(x):
        return savgol_filter(x / (2 * np.pi), range, order, deriv=1, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def cryoscope_frequency(ds, sg_range=3, sg_order=2):
    """Extract cryoscope frequency from unwrapped phase.

    Mean-detuning derivative f = d(phi/2pi)/dt following Rol et al. arXiv:1907.04818,
    Eq. (3); the derivative is obtained with a Savitzky-Golay filter. Flux conversion
    is left to the caller (``_compute_flux_response`` / coupler dispersion).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Unwrapped phase with a ``time`` dimension (and optional ``qubit``).
    sg_range, sg_order : int
        Savitzky-Golay window length and polynomial order.

    Returns
    -------
    xr.Dataset
        Dataset with a ``"freq"`` variable (GHz detuning convention).
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.copy().to_dataset(name="phase")
        _phase = ds["phase"]
    else:
        ds = ds.copy()
        _phase = ds[list(ds.data_vars)[0]]

    ds["freq"] = diff_savgol(_phase, "time", range=sg_range, order=sg_order)
    return ds


# --- Freq → flux and multi-exp packaging (mirrors long-distortion) ---


def _compute_flux_response(
    cryoscope_freq: xr.DataArray,
    qubits: list,
    dim_names: List[str],
    node: QualibrationNode,
    freq_to_flux_source: FreqFluxSource = "auto",
) -> Tuple[xr.DataArray, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, str]]:
    """Map cryoscope frequency (GHz) to a Z-flux step response magnitude.

    Same ``resolve_freq_flux_curve`` decision as long-distortion / pre-run amplitude
    selection. Cryoscope convention: ``freq = f_drive - f_qubit`` (GHz), so
    absolute qubit frequency is ``RF - freq * 1e9``.
    """
    measured_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sources: Dict[str, str] = {}
    flux_response = xr.DataArray(
        np.full_like(cryoscope_freq.values, np.nan, dtype=float),
        coords=cryoscope_freq.coords,
        dims=cryoscope_freq.dims,
    )

    for i, q in enumerate(qubits):
        if i >= len(dim_names):
            break
        dim_name = dim_names[i]
        selected = resolve_freq_flux_curve(q, freq_to_flux_source)
        sources[q.name] = selected.label
        freq_ghz = cryoscope_freq.sel(qubit=dim_name).values
        abs_freq_q = q.xy.RF_frequency - freq_ghz * 1e9

        if selected.is_measured:
            curve = selected.curve
            measured_curves[q.name] = curve
            flux_response.values[i, :] = frequency_to_flux_deviation(
                abs_freq_q,
                curve[0],
                curve[1],
                q.xy.RF_frequency,
            )
        elif selected.quad_term is not None:
            flux_response.values[i, :] = np.sqrt(np.abs(freq_ghz * 1e9 / selected.quad_term))
        else:
            node.log(
                f"  WARNING: {q.name}: no freq-vs-flux relation available "
                f"(freq_to_flux_source='{freq_to_flux_source}'); flux response left as NaN."
            )

    return flux_response, measured_curves, sources


def _attach_spec_curve_vars(
    ds: xr.Dataset,
    spec_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> xr.Dataset:
    """Attach the measured freq-vs-flux curves used (pad to a common length with NaN)."""
    if not spec_curves:
        return ds
    qubit_names_sc = list(spec_curves.keys())
    n_pts = max(len(spec_curves[qn][0]) for qn in qubit_names_sc)

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(n_pts, np.nan, dtype=float)
        out[: len(arr)] = arr
        return out

    ds = ds.copy()
    ds["spec_curve_flux"] = xr.DataArray(
        np.array([_pad(spec_curves[qn][0]) for qn in qubit_names_sc]),
        dims=["spec_qubit", "spec_pts"],
        coords={"spec_qubit": qubit_names_sc},
    )
    ds["spec_curve_freq"] = xr.DataArray(
        np.array([_pad(spec_curves[qn][1]) for qn in qubit_names_sc]),
        dims=["spec_qubit", "spec_pts"],
        coords={"spec_qubit": qubit_names_sc},
    )
    return ds


def _extract_relevant_fit_parameters(
    ds: xr.Dataset,
    node: QualibrationNode,
    dim_names: Optional[List[str]] = None,
) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the flux step response per dataset qubit and package ``FitParameters``.

    Mirrors the long-distortion helper of the same name. Keys follow the dataset
    ``qubit`` coordinate (qubit names).
    """
    n_exponentials = int(node.parameters.n_exponentials)
    flux = ds["flux_response"]
    names = dim_names if dim_names is not None else [str(v) for v in flux.qubit.values]
    time_vals = np.asarray(ds.time.values, dtype=float)
    fit_results: Dict[str, FitParameters] = {}

    for dim_name in names:
        flux_vals = np.asarray(flux.sel(qubit=dim_name).values, dtype=float)
        mask = np.isfinite(flux_vals) & (time_vals > 0)
        if mask.sum() < max(2 * n_exponentials + 1, 4):
            fit_results[dim_name] = FitParameters(
                success=False,
                n_components_requested=n_exponentials,
                n_components_used=0,
                a_tau_tuple=[],
                a_dc=float("nan"),
                rms_error=float("nan"),
            )
            continue
        fit_results[dim_name] = multi_exp_fit_global(time_vals[mask], flux_vals[mask], n_exponentials, verbose=True)
    return ds, fit_results


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """Fit raw cryoscope data with exponential models for each qubit.

    Pipeline: oscillation fit → unwrap → cryoscope freq → shared freq→flux
    (``flux_response``) → ``multi_exp_fit_global``.

    Returns
    -------
    ds_fit : xr.Dataset
        Dataset with ``"freq"`` and ``"flux_response"`` (and optional ``spec_curve_*``).
    fit_results : dict[str, FitParameters]
        Shared ``FitParameters`` (``a_tau_tuple``, ``a_dc``, ``rms_error``, …),
        keyed by the dataset ``qubit`` coordinate.
    """
    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise ValueError("Dataset must contain either 'I' or 'state' data")

    dafit = fit_oscillation(ds[data], "frame")
    daphi = unwrap_phase(dafit.sel(fit_vals="phi"), "time")

    qubits = node.namespace["qubits"]
    dim_names = [str(v) for v in ds[data].qubit.values]
    ds_fit = cryoscope_frequency(daphi, sg_order=2, sg_range=3)

    freq_to_flux_source: FreqFluxSource = getattr(node.parameters, "freq_to_flux_source", "auto")
    flux_response, measured_curves, sources = _compute_flux_response(
        ds_fit["freq"],
        qubits,
        dim_names,
        node,
        freq_to_flux_source=freq_to_flux_source,
    )
    ds_fit["flux_response"] = flux_response
    for qname, label in sources.items():
        node.log(f"  {qname}: freq -> flux conversion used {label}")
    ds_fit = _attach_spec_curve_vars(ds_fit, measured_curves)
    ds_fit.attrs["freq_to_flux_source"] = str(freq_to_flux_source)
    ds_fit.attrs["freq_to_flux_sources"] = "; ".join(f"{k}: {v}" for k, v in sources.items())

    return _extract_relevant_fit_parameters(ds_fit, node, dim_names=dim_names)


# --- FIR analysis (short-distortion only) ---


def fit_fir_data(ds_fit: xr.Dataset, node) -> dict:
    """Run FIR filter analysis on the cryoscope flux step response.

    Pipeline:
      1. Normalize flux by stable-region tail mean.
      2. Resample from 1 GS/s to 2 GS/s on ``ds_fit.time``.
      3. Fit forward FIR of length ``fir_max_taps`` and invert for feedforward.
      4. Validate corrected response at 1 GS/s.
    """
    from calibration_utils.common_utils.flux_distortions.fir_utils import (
        analyze_inverse_fir,
        estimate_noise_floor,
        resample_to_target_rate,
    )
    from scipy.signal import lfilter

    params = node.parameters
    dim_names = [str(v) for v in ds_fit.flux_response.qubit.values]
    fir_results = {}

    for dim_name in dim_names:
        flux_raw = ds_fit.flux_response.sel(qubit=dim_name).values
        if np.all(np.isnan(flux_raw)):
            node.log(f"  {dim_name}: flux is all NaN — skipping FIR")
            fir_results[dim_name] = {"success": False}
            continue

        tail_mean = float(np.nanmean(flux_raw[-10:]))
        if tail_mean == 0:
            tail_mean = 1.0
        normalized_1gs = flux_raw / tail_mean

        time_1gs_arr = np.asarray(ds_fit.time.values, dtype=float)
        normalized_2gs, time_2gs = resample_to_target_rate(
            normalized_1gs,
            original_Ts=1,
            target_Ts=0.5,
            t_original_ns=time_1gs_arr,
        )

        h_fir, h_inv, _reconstructed, fir_info = analyze_inverse_fir(
            response=normalized_2gs,
            Ts=0.5,
            L=params.fir_max_taps,
        )

        ideal_1gs = np.ones(len(normalized_1gs))
        predistorted = lfilter(h_inv, 1, ideal_1gs)
        corrected = lfilter(h_fir, 1, predistorted)
        corrected_norm = corrected / float(np.nanmean(corrected[-10:]))

        noise_info = estimate_noise_floor(normalized_1gs, Ts=1.0)

        fir_results[dim_name] = {
            "success": True,
            "forward_fir": h_fir.tolist(),
            "inverse_fir": h_inv.tolist(),
            "normalized_1gs": normalized_1gs.tolist(),
            "corrected_1gs": corrected_norm.tolist(),
            "time_1gs": ds_fit.time.values.tolist(),
            "time_2gs": time_2gs.tolist(),
            "normalized_2gs": normalized_2gs.tolist(),
            "L": fir_info["L"],
            "lam": fir_info["lam"],
            "lam_smooth": fir_info["lam_smooth"],
            "sigma_ns": fir_info["sigma_ns"],
            "forward_nrms": fir_info["forward_nrms"],
            "noise_sigma_A_tail_std": noise_info["sigma_A"],
            "noise_sigma_B_first_diff": noise_info["sigma_B"],
            "noise_ratio_AB": noise_info["ratio_AB"],
            "noise_estimate_status": noise_info["status"],
            "noise_estimate_msg": noise_info["msg_short"],
        }
        node.log(
            f"  {dim_name}: FIR done — L={fir_info['L']}, "
            f"NRMS={fir_info['forward_nrms']:.3e}, "
            f"forward {len(h_fir)} taps, inverse {len(h_inv)} taps"
        )
        node.log(
            f"  {dim_name}: noise sigma_A={noise_info['sigma_A']:.2e} "
            f"sigma_B={noise_info['sigma_B']:.2e} "
            f"ratio_AB={noise_info['ratio_AB']:.2f} -> {noise_info['msg_short']}"
        )
        if noise_info["status"] == "warn_tail":
            node.log(f"  {dim_name}: tail may not be settled (sigma_A >> sigma_B); " f"try a longer cryoscope_len.")

    return fir_results
