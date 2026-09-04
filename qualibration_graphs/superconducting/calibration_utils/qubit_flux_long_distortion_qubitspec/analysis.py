"""Analysis utilities for π vs flux (qubitspec) long flux-distortion characterization.

Extracts the instantaneous qubit frequency vs flux-pulse duration from a 2-D
spectroscopy map, converts frequency to a Z-flux step response via a
Ramsey / spectroscopy / quad_term cascade, then fits a sum of decaying
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

2. Frequency → flux uses the relation named by ``freq_to_flux_source``. With
   the default ``"auto"``:

       Ramsey vs Z (09a)  →  spectroscopy vs Z (03b)  →  quad_term fallback

       |Δf| ≈ q · Φ²     (near idle; quad_term path)
       |ΔΦ|(t) = frequency_to_flux_deviation(f(t); curve)

   The qubit is assumed to be parked at its sweetspot, so ``f(Φ)`` is symmetric
   about idle and the side of the parabola is not a user choice; the response is
   returned as a magnitude, which is what the taps ``A_i = a_i / a_dc`` need.

3. The flux step response is the deviation from idle bias. Late-time asymptote
   is a_dc (the commanded long-pulse amplitude in flux units).

4. Multi-exponential model of the step response:

       y(t) = a_dc + sum_i a_i exp(-t / tau_i)

   Fitted with ``multi_exp_fit_global``. IIR taps follow
   A_i = a_i / a_dc (Rol et al. arXiv:1907.04818 Eq. (S22)).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import xarray as xr
from calibration_utils.common_utils.flux_distortions import (
    FitParameters,
    multi_exp_fit_global,
)
from calibration_utils.common_utils.flux_distortions.curves import (
    FreqFluxSource,
    frequency_to_flux_deviation,
    resolve_freq_flux_curve,
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
            log_callable(f"{qb}: fit FAILED — see warnings above (spectroscopy edge or freq→flux curve).")


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
    log_callable: Callable[[str], None],
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
                log_callable(f"  WARNING: {label}: no clear I/Q resonance; using {best_name} anyway.")
            else:
                log_callable(f"  {label}: center-frequency extraction used {best_name}.")
        center_freqs[i] = traces[best_name]

    # Soft edge check: clipped f(t) vs detuning axis (warn only).
    f_lo, f_hi = float(np.min(freqs)), float(np.max(freqs))
    tol = 0.02 * (f_hi - f_lo)
    for i in range(n_qubits):
        label = qubit_names[i] if i < len(qubit_names) else f"qubit {i}"
        finite = center_freqs[i][np.isfinite(center_freqs[i])]
        if finite.size == 0:
            log_callable(f"  WARNING: {label}: no resonance found in any time slice.")
            continue
        n_edge = int(((finite <= f_lo + tol) | (finite >= f_hi - tol)).sum())
        if n_edge:
            lo_mhz, hi_mhz = f_lo / 1e6, f_hi / 1e6
            log_callable(
                f"  WARNING: {label}: resonance on spectroscopy sweep edge ({n_edge}/{finite.size} slices). "
                f"In 17a, increase frequency_span_in_mhz or detuning_in_mhz "
                f"(current detuning sweep: {lo_mhz:.0f} to {hi_mhz:.0f} MHz vs RF)."
            )

    template = ds[candidates[0]].transpose("qubit", "time", freq_dim).isel({freq_dim: 0}, drop=True)
    return xr.DataArray(center_freqs, dims=template.dims, coords=template.coords).rename("center_frequency")


def _compute_flux_response(
    center_freqs: xr.DataArray,
    qubits: list,
    freq_to_flux_source: FreqFluxSource = "auto",
    *,
    log_callable: Callable[[str], None],
) -> Tuple[xr.DataArray, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, str]]:
    """Map extracted ``center_freqs(t)`` to a Z-flux step response magnitude.

    The freq→voltage relation is chosen by ``resolve_freq_flux_curve`` — the
    single decision point also used to pick the Z amplitude before the run — so
    ``freq_to_flux_source`` is the only knob. With ``"auto"`` the order is
    Ramsey vs flux (09a) → spectroscopy vs flux (03b) → quadratic
    ``freq_vs_flux_01_quad_term``, with run IDs taken from ``qubit.extras``.

    Returns
    -------
    (flux_response, measured_curves, sources)
        ``measured_curves`` holds the measured curve per qubit (for the debug
        figure); ``sources`` maps qubit name → the relation actually used, so it
        can be logged and shown on the figures.
    """
    measured_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sources: Dict[str, str] = {}
    flux_response = xr.DataArray(
        np.full_like(center_freqs.values, np.nan),
        coords=center_freqs.coords,
        dims=center_freqs.dims,
    )

    for i, q in enumerate(qubits):
        selected = resolve_freq_flux_curve(q, freq_to_flux_source)
        sources[q.name] = selected.label

        if selected.is_measured:
            curve = selected.curve
            measured_curves[q.name] = curve
            flux_c, freq_c = curve
            freq_span_mhz = (float(np.max(freq_c)) - float(np.min(freq_c))) / 1e6
            if freq_span_mhz < 10.0:
                cal_node = "09a" if selected.kind == "ramsey" else "03b"
                log_callable(
                    f"  WARNING: {q.name}: {selected.label} spans only {freq_span_mhz:.2f} MHz in frequency "
                    f"({np.min(freq_c)/1e9:.3f}–{np.max(freq_c)/1e9:.3f} GHz). "
                    f"Re-run {cal_node} with flux_span ≥ 0.2 V and save_load_id=True."
                )
            abs_freq_q = center_freqs.sel(qubit=q.name).values + q.xy.RF_frequency
            y = frequency_to_flux_deviation(
                abs_freq_q,
                curve[0],
                curve[1],
                q.xy.RF_frequency,
            )
            flux_response.values[i, :] = y
            if not np.isfinite(y).any():
                f_meas = abs_freq_q[np.isfinite(abs_freq_q)]
                f_lo = float(np.min(f_meas)) / 1e9 if f_meas.size else float("nan")
                f_hi = float(np.max(f_meas)) / 1e9 if f_meas.size else float("nan")
                log_callable(
                    f"  WARNING: {q.name}: flux_response is all NaN — f(t) is {f_lo:.3f}–{f_hi:.3f} GHz but "
                    f"{selected.label} only covers {np.min(freq_c)/1e9:.3f}–{np.max(freq_c)/1e9:.3f} GHz. "
                    f"Re-run 09a/03b with wider flux_span."
                )
        elif selected.quad_term is not None:
            flux_response.values[i, :] = np.sqrt(np.abs(center_freqs.sel(qubit=q.name).values / selected.quad_term))
        else:
            log_callable(
                f"  WARNING: {q.name}: no freq→flux map (set freq_to_flux_source or run 09a/03b with save_load_id)."
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
        log_callable=node.log,
    )

    freq_to_flux_source = getattr(node.parameters, "freq_to_flux_source", "auto")
    flux_response, spec_curves, sources = _compute_flux_response(
        center_freqs,
        qubits,
        freq_to_flux_source=freq_to_flux_source,
        log_callable=node.log,
    )

    for qname, label in sources.items():
        node.log(f"  {qname}: freq→flux via {label}")

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response
    ds = _attach_spec_curve_vars(ds, spec_curves)
    ds.attrs["freq_to_flux_source"] = str(freq_to_flux_source)
    ds.attrs["freq_to_flux_sources"] = "; ".join(f"{k}: {v}" for k, v in sources.items())

    return _extract_relevant_fit_parameters(ds, node)
