"""Analysis utilities for π vs flux (qubitspec) long flux-distortion characterization.

Extracts the instantaneous qubit frequency vs flux-pulse duration from a 2-D
spectroscopy map, converts frequency to a Z-flux step response via a
spectroscopy / Ramsey / quad_term cascade, then fits a sum of decaying
exponentials for IIR predistortion taps.

Signal -> flux response pipeline:

    flowchart TD
        rawIQ["raw I/Q or state vs (time, detuning)"]
            --> center["Gaussian peak/dip fit per time slice\n  -> center_freqs(t)"]
        center --> invert["freq → flux deviation\n  (spec curve / Ramsey curve / quad_term)"]
        invert --> fluxResp["flux_response(t) = Φ(t) - Φ_idle"]
        fluxResp --> fit["multi_exp_fit_global\n  -> a_dc, {(a_i, tau_i)}"]
        fit --> iir["IIR: A_i = a_i / a_dc, tau_i unchanged"]

Key equations
-------------
1. At each flux-pulse duration t, a π (or chosen XY) pulse probes the qubit
   while the XY drive is swept in detuning. The resonance center f(t) is the
   Gaussian peak (or dip) along the frequency axis of that time slice.

2. Frequency → flux uses the first available freq↔Z curve, restricted to the
   chosen ``flux_branch`` (left/right of the idle sweet spot):

       spectroscopy vs Z  →  Ramsey vs Z (09a)  →  quad_term fallback

       |Δf| ≈ q · Φ²     (near idle; quad_term path)
       Φ(t) = frequency_to_flux_deviation(f(t); curve, branch)

3. The flux step response is the deviation from idle bias. Late-time asymptote
   is a_dc (the commanded long-pulse amplitude in flux units).

4. Multi-exponential model of the step response:

       y(t) = a_dc + sum_i a_i exp(-t / tau_i)

   Fitted with ``multi_exp_fit_global``. IIR taps follow
   A_i = a_i / a_dc (Rol et al. arXiv:1907.04818 Eq. (S22)).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import xarray as xr
from calibration_utils.common_utils.flux_distortions import (
    FitParameters,
    multi_exp_fit_global,
)
from calibration_utils.common_utils.flux_distortions.curves import (
    frequency_to_flux_deviation,
    load_ramsey_curve,
    load_spectroscopy_curve,
)
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit


# --- Dataset preprocessing and center-frequency extraction ---
def log_fitted_results(fit_results: Dict, log_callable=print) -> None:
    """Log the fitted pi vs flux results for each qubit.

    Expects dict-form results (after ``asdict``), matching qubit spectroscopy.
    """
    # Surfaces fit quality and key parameters to the console so operators can spot failed fits before updating state.
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


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V, add amplitude/phase, and attach freq/flux axis coords."""
    # Normalises the raw hardware output into calibrated voltage units and attaches derived coordinates needed by downstream analysis and plotting.
    qubits = node.namespace["qubits"]
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, qubits)  # type: ignore
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace and node.namespace["sweep_axes"].get("detuning") is not None
        else np.asarray(ds["detuning"].values, dtype=float)
    )
    # Plotting uses freq_full; flux is a quadratic placeholder for axis labels only.
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency for q in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.array(
                    [
                        (
                            np.sqrt(np.maximum(0.0, dfs / q.freq_vs_flux_01_quad_term))
                            if getattr(q, "freq_vs_flux_01_quad_term", None) and q.freq_vs_flux_01_quad_term != 0
                            else np.full_like(dfs, np.nan, dtype=float)
                        )
                        for q in qubits
                    ]
                ),
            ),
        }
    )
    ds.freq_full.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for peak fitting."""
    # Provides the model curve used to locate the qubit resonance peak within each spectroscopy frequency slice.
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian(freqs, states, polarity: float = 1.0):
    """Fit one 1-D spectroscopy slice and return the resonance frequency.

    ``freqs`` / ``states`` are signal vs drive frequency at a fixed flux-pulse
    time. ``polarity`` is ``+1`` (peak) or ``-1`` (dip): raw I/Q can show either
    depending on readout rotation; ``state`` is usually a peak.
    """
    freqs = np.asarray(freqs, dtype=float)
    states = np.asarray(states, dtype=float)
    if not np.isfinite(states).any():
        return float("nan")

    f_lo, f_hi = float(np.min(freqs)), float(np.max(freqs))
    span = f_hi - f_lo
    y_min, y_max = float(np.min(states)), float(np.max(states))
    swing = y_max - y_min
    if polarity >= 0:
        a0, x0, offset0, a_lo, a_hi = swing, float(freqs[np.argmax(states)]), y_min, 0.0, np.inf
    else:
        a0, x0, offset0, a_lo, a_hi = -swing, float(freqs[np.argmin(states)]), y_max, -np.inf, 0.0

    # Bounding x0 to the swept range keeps a failed slice at the sweep edge instead of letting
    # the optimiser wander outside it and report a frequency that was never measured.
    lower = [a_lo, f_lo, span / len(freqs), -np.inf]
    upper = [a_hi, f_hi, span, np.inf]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip([a0, x0, span / 10, offset0], lower, upper)]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0, bounds=(lower, upper))
        return float(popt[1])
    except Exception:
        return float("nan")


def _fit_center_freq_map(values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Turn one qubit's 2-D spectroscopy map into ``center_freq(t)``.

    ``values`` has shape ``(n_times, n_freqs)``: each row is a 1-D qubit
    spectroscopy (signal vs frequency) at one flux-pulse duration. Rows are
    stacked over time while the Z pulse is on.

    Peak vs dip (``polarity``) is fixed once for the whole map: subtract each
    row's median to remove baseline, then compare the strongest positive vs
    negative excursion. I/Q polarity follows readout rotation and is constant
    for a run; choosing it per row would flip with noise. Every row is then
    Gaussian-fitted with that same sign.
    """
    # Baseline-free deviations: median per time row ≈ off-resonant level.
    deviation = values - np.nanmedian(values, axis=-1, keepdims=True)
    polarity = 1.0 if abs(np.nanmax(deviation)) >= abs(np.nanmin(deviation)) else -1.0
    return np.array([fit_gaussian(freqs, row, polarity) for row in values])


def _trace_roughness(center_freqs: np.ndarray) -> float:
    """Score how jagged an extracted ``f(t)`` trace is (lower is better).

    When fitting raw IQ, readout rotation may put the resonance on I or on Q.
    The true flux step response is smooth in time, so the quadrature that
    carries the signal yields a smoother ``f(t)`` than the noisy one.
    ``extract_center_freqs`` fits both and keeps the smaller roughness.

    Defined as ``std(diff(f)) / (sqrt(2) * ptp(f))`` over finite points; returns
    ``inf`` if the trace is too short or flat to rank.
    """
    finite = center_freqs[np.isfinite(center_freqs)]
    if finite.size < 10:
        return np.inf
    excursion = float(np.ptp(finite))
    if excursion <= 0:
        return np.inf
    return float(np.std(np.diff(finite)) / np.sqrt(2) / excursion)


def extract_center_freqs(
    ds: xr.Dataset,
    freqs: np.ndarray,
    *,
    use_state_discrimination: bool = False,
) -> xr.DataArray:
    """Extract resonance center frequencies vs time from a 2-D spectroscopy map.

    Per qubit and time slice, fits a Gaussian along the frequency axis. With state
    discrimination, uses ``state``. Otherwise fits ``I`` and ``Q`` (or ``IQ_abs``)
    and keeps the smoother trace — readout rotation decides which quadrature carries
    the resonance, and |IQ| would fold in the empty-quadrature noise.

    Also logs a soft warning if many extracted ``f(t)`` points sit at the edges of
    the detuning sweep (``freqs``), which usually means the span is too narrow.
    """
    freq_dim = "detuning" if "detuning" in ds.dims else "freq"

    if use_state_discrimination and "state" in ds.data_vars:
        candidates = ["state"]
    else:
        candidates = [name for name in ("I", "Q") if name in ds.data_vars]
        if not candidates:
            if "IQ_abs" not in ds.data_vars:
                raise ValueError("Dataset is missing state, I/Q, and IQ_abs for center-freq extraction")
            candidates = ["IQ_abs"]

    stacked = {
        name: np.asarray(ds[name].transpose("qubit", "time", freq_dim).values, dtype=float) for name in candidates
    }
    qubit_names = list(np.atleast_1d(ds["qubit"].values)) if "qubit" in ds.coords else []
    n_qubits, n_times = next(iter(stacked.values())).shape[:2]

    center_freqs = np.full((n_qubits, n_times), np.nan)
    for i in range(n_qubits):
        label = qubit_names[i] if i < len(qubit_names) else f"qubit {i}"
        traces = {name: _fit_center_freq_map(stacked[name][i], freqs) for name in candidates}
        if len(candidates) == 1:
            best_name = candidates[0]
        else:
            ranked = sorted(candidates, key=lambda name: _trace_roughness(traces[name]))
            best_name = ranked[0]
            if not np.isfinite(_trace_roughness(traces[best_name])):
                print(f"  WARNING: {label}: no usable resonance in any quadrature; " f"falling back to {best_name}.")
            else:
                print(f"  {label}: center-frequency extraction used the {best_name} quadrature.")
        center_freqs[i] = traces[best_name]

    # Soft edge check: clipped f(t) vs detuning axis (warn only).
    f_lo, f_hi = float(np.min(freqs)), float(np.max(freqs))
    tol = 0.02 * (f_hi - f_lo)
    for i in range(n_qubits):
        label = qubit_names[i] if i < len(qubit_names) else f"qubit {i}"
        finite = center_freqs[i][np.isfinite(center_freqs[i])]
        if finite.size == 0:
            print(f"  WARNING: {label}: no resonance could be extracted from any time slice.")
            continue
        n_edge = int(((finite <= f_lo + tol) | (finite >= f_hi - tol)).sum())
        if n_edge:
            print(
                f"  WARNING: {label}: {n_edge}/{finite.size} extracted resonances sit at the "
                f"edge of the detuning sweep. Widen frequency_span_in_mhz or re-centre "
                f"detuning_in_mhz; continuing with the clipped estimates."
            )

    template = ds[candidates[0]].transpose("qubit", "time", freq_dim).isel({freq_dim: 0}, drop=True)
    return xr.DataArray(center_freqs, dims=template.dims, coords=template.coords).rename("center_frequency")


def _compute_flux_response(
    center_freqs: xr.DataArray,
    qubits: list,
    use_spec: bool,
    spec_run_id: Optional[int],
    use_ramsey: bool = False,
    ramsey_run_id: Optional[int] = None,
    flux_amp_for_detuning: Optional[float] = None,
    flux_branch: Literal["left", "right"] = "right",
) -> Tuple[xr.DataArray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Map extracted ``center_freqs(t)`` to a Z-flux step response.

    Per qubit, invert frequency via the first available freq↔flux curve:
    1. Spectroscopy vs Z-flux (``use_spec`` + ``spec_run_id``)
    2. Ramsey vs Z-flux / 09a (``use_ramsey`` + run id / extras)
    3. Quadratic ``freq_vs_flux_01_quad_term`` from QUAM
    """
    spec_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    flux_response = xr.DataArray(
        np.full_like(center_freqs.values, np.nan),
        coords=center_freqs.coords,
        dims=center_freqs.dims,
    )

    for i, q in enumerate(qubits):
        curve = None
        # Path 1: spectroscopy vs Z-flux
        if use_spec and spec_run_id is not None:
            curve = load_spectroscopy_curve(spec_run_id, q.name, q.xy.RF_frequency)
            if curve is not None:
                spec_curves[q.name] = curve
        # Path 2: Ramsey vs Z-flux (global override or per-qubit extras)
        if curve is None and use_ramsey:
            curve = load_ramsey_curve(q, ramsey_run_id)
        # Path 3: quad_term fallback
        if curve is not None:
            abs_freq_q = center_freqs.sel(qubit=q.name).values + q.xy.RF_frequency
            # Branch: namespace sentinel/amp vs idle flux when available, else flux_branch param.
            use_upper_branch = flux_branch == "right"
            if flux_amp_for_detuning is not None:
                idle_idx = int(np.argmin(np.abs(curve[1] - q.xy.RF_frequency)))
                use_upper_branch = float(flux_amp_for_detuning) >= float(curve[0][idle_idx])
            flux_response.values[i, :] = frequency_to_flux_deviation(
                abs_freq_q,
                curve[0],
                curve[1],
                q.xy.RF_frequency,
                use_upper_branch=use_upper_branch,
            )
        else:
            qt = getattr(q, "freq_vs_flux_01_quad_term", None) or np.nan
            if np.isfinite(qt) and qt != 0:
                sign = 1.0 if flux_branch == "right" else -1.0
                flux_response.values[i, :] = sign * np.sqrt(np.abs(center_freqs.sel(qubit=q.name).values / qt))

    return flux_response, spec_curves


def _attach_spec_curve_vars(
    ds: xr.Dataset,
    spec_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spec_run_id: Optional[int],
) -> xr.Dataset:
    """Attach spectroscopy calibration curves (pad to a common length with NaN)."""
    if not spec_curves:
        return ds
    qubit_names_sc = list(spec_curves.keys())
    n_pts = max(len(spec_curves[qn][0]) for qn in qubit_names_sc)

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(n_pts, np.nan, dtype=float)
        out[: len(arr)] = arr
        return out

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
    ds.attrs["spectroscopy_run_id"] = int(spec_run_id)
    return ds


def _extract_relevant_fit_parameters(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the flux step response per qubit and package ``FitParameters``.

    Mirrors the qubit-spectroscopy helper of the same name: after the dataset has
    been enriched with analysis products (here ``flux_response``), run the node
    fit and package per-qubit results for logging / state update.
    """
    qubits = node.namespace["qubits"]
    n_exponentials = int(node.parameters.n_exponentials)
    flux_response = ds["flux_response"]
    fit_results: Dict[str, FitParameters] = {}
    for q in qubits:
        qf = flux_response.sel(qubit=q.name)
        t_data = np.asarray(qf.time.values, dtype=float)
        y_data = np.asarray(qf.values, dtype=float)
        mask = np.isfinite(y_data) & (t_data > 0)
        if mask.sum() < max(2 * n_exponentials + 1, 4):
            fit_results[q.name] = FitParameters(
                success=False,
                n_components_requested=n_exponentials,
                n_components_used=0,
                a_tau_tuple=[],
                a_dc=float("nan"),
                rms_error=float("nan"),
            )
            continue
        fit_results[q.name] = multi_exp_fit_global(t_data[mask], y_data[mask], n_exponentials, verbose=True)
    return ds, fit_results


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract center frequencies, map to flux response, and fit exponential cascade.

    Returns ``(ds_fit, fit_results)`` like qubit spectroscopy: the enriched dataset
    (``center_freqs``, ``flux_response``, …) plus per-qubit ``FitParameters``.
    """
    # Top-level analysis entry point called by the calibration node: extracts resonance frequencies, maps them to flux, and fits the exponential distortion model for each qubit.
    qubits = node.namespace["qubits"]

    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else ds["detuning"].values
    )

    center_freqs = extract_center_freqs(
        ds,
        dfs,
        use_state_discrimination=bool(node.parameters.use_state_discrimination and "state" in ds.data_vars),
    )

    spec_run_id = getattr(node.parameters, "spectroscopy_run_id", None)
    use_spec = getattr(node.parameters, "use_spectroscopy_data", False)
    use_ramsey = getattr(node.parameters, "use_ramsey_data", False)
    ramsey_run_id = getattr(node.parameters, "ramsey_run_id", None)
    flux_amp_for_detuning = node.namespace.get("flux_amp_for_detuning")
    flux_response, spec_curves = _compute_flux_response(
        center_freqs,
        qubits,
        use_spec,
        spec_run_id,
        use_ramsey=use_ramsey,
        ramsey_run_id=ramsey_run_id,
        flux_amp_for_detuning=flux_amp_for_detuning,
        flux_branch=getattr(node.parameters, "flux_branch", "right"),
    )

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response
    ds = _attach_spec_curve_vars(ds, spec_curves, spec_run_id)

    return _extract_relevant_fit_parameters(ds, node)
