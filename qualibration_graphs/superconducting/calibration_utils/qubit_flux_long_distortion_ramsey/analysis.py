"""Analysis utilities for Ramsey-based qubit flux long distortion characterization.

Maps the measured Ramsey phase at each delay time to an effective qubit flux
amplitude using a reference Ramsey amplitude sweep, then fits a sum of decaying
exponentials to the residual flux response.

Signal -> flux response pipeline:

    flowchart TD
        rawIQ["raw IQ vs frame for each t_delay"]
            --> phase["Fourier projection phi(t)=atan2(-sin_proj, cos_proj)"]
        refSweep["reference IQ vs frame for each ramsey_flux_amplitude"]
            --> refPhase["phi_ref(A_flux) (unwrapped along amplitude)"]
        phase --> invert["per-t inverse interp:\n  eff_amp(t) = phi_ref^-1(phi(t))"]
        refPhase --> invert
        invert --> distortion["residual flux:\n  delta(t) = eff_amp(t) - ramsey_flux_amp"]
        distortion --> stepResp["step-rise reformulation:\n  y(t) = qubit_flux_amp - delta(t)"]
        stepResp --> fit["multi_exp_fit_global(t_pulse_ns=flux_settle_time_in_ns)\n  -> a_dc, {(a_i, tau_i)}"]
        fit --> iir["IIR: A_i = a_i / a_dc, tau_i unchanged"]

Key equations
-------------
1. Ramsey phase accumulation during the wait window of length T_wait at
   instantaneous effective amplitude A_eff(t):

       phi(t_delay) = 2 pi * Delta_f(A_eff(t_delay)) * T_wait

2. Reference calibration (no preceding long pulse) gives phi_ref(A_flux);
   invert via 1-D interpolation to map measured phi -> A_eff. The reference is
   monotonised in amplitude first (np.interp needs an increasing abscissa and a
   phase-sort alone would scramble the amplitude correspondence), and phases
   falling outside the reference window are dropped rather than clamped to its
   endpoints.

3. Step-rise reformulation: a positive residual delta(t) (long-pulse tail
   that has not yet decayed) is recast as the equivalent step-rise response
   y(t) = qubit_flux_amp - delta(t). So at t = 0+ (immediately after the
   long pulse is turned off) y ~ 0, and at t -> infinity y -> qubit_flux_amp
   = a_dc.

4. Finite-pulse multi-exponential fit
   [Aggarwal et al. arXiv:2503.08645 Appendix H, Eq. (H1), adapted]:

       y(t_delay) = a_dc + sum_i a_i (1 - exp(-T_pulse/tau_i)) exp(-t_delay/tau_i)

   The multi_exp_fit_global(..., t_pulse_ns=T_pulse) call returns
   de-attenuated amplitudes a_i, so the IIR coefficient formula
   A_i = a_i / a_dc (gain-normalized IIR tap; Rol et al. arXiv:1907.04818 Eq. (S22),
   s(t)=g(1+A e^{-t/tau_IIR})u(t), with g <-> a_dc and A <-> a_i/a_dc) yields the
   correct per-pole pre-distortion strength directly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from calibration_utils.common_utils.flux_distortions import (
    FitParameters,
    multi_exp_fit_global,
)
from qualibration_libs.data import convert_IQ_to_V

# --- Dataset preprocessing ---


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Preprocess Ramsey raw dataset: convert IQ to volts if applicable."""
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


# --- Phase extraction ---


def _fourier_phase(data: xr.DataArray, dim: str = "frame") -> xr.DataArray:
    """Oscillation phase via Fourier projection at the fundamental frequency."""
    coord = data.coords[dim]
    cos_basis = xr.DataArray(np.cos(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})
    sin_basis = xr.DataArray(np.sin(2 * np.pi * coord.values), dims=[dim], coords={dim: coord})
    centered = data - data.mean(dim=dim)
    return xr.apply_ufunc(
        np.arctan2,
        -(centered * sin_basis).sum(dim=dim),
        (centered * cos_basis).sum(dim=dim),
    )


def _robust_unwrap_1d(phase):
    """Unwrap 1-D phase with linear-trend prediction (handles >π steps on log axes)."""
    period = 2 * np.pi
    out = np.array(phase, dtype=float)
    for i in range(1, len(out)):
        predicted = out[i - 1] if i < 2 else 2 * out[i - 1] - out[i - 2]
        out[i] -= np.round((out[i] - predicted) / period) * period
    return out


def extract_phases(ds: xr.Dataset) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """Extract signal phase φ(t) and optional reference calibration φ_ref(a).

    Signal phase stays wrapped in [-π, π] (per-point inversion). Reference phase
    is unwrapped along amplitude. Uses ``state``/``state_ref`` or ``I``/``I_ref``.
    """
    if "state" in ds.data_vars:
        signal_key, ref_key = "state", ("state_ref" if "state_ref" in ds.data_vars else None)
    elif "I" in ds.data_vars:
        signal_key, ref_key = "I", ("I_ref" if "I_ref" in ds.data_vars else None)
    else:
        raise ValueError("Dataset must contain 'state' or 'I' data variable")

    signal_phase = _fourier_phase(ds[signal_key], "frame")

    ref_cal = None
    if ref_key is not None and ref_key in ds.data_vars and "a" in ds[ref_key].dims:
        ref_cal = xr.apply_ufunc(
            _robust_unwrap_1d,
            _fourier_phase(ds[ref_key], "frame"),
            input_core_dims=[["a"]],
            output_core_dims=[["a"]],
            vectorize=True,
        )
    return signal_phase, ref_cal


# --- Phase → flux response ---


def _monotonic_reference(ref_amps: np.ndarray, ref_phases: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Build a strictly-increasing-in-phase reference curve for ``np.interp``.

    ``np.interp`` requires an increasing abscissa, so the reference has to be
    ordered by phase. Sorting the phases alone is not enough: if φ_ref(a) is not
    monotonic in amplitude (noise, or a span approaching 2π), sorting scrambles
    the amplitude correspondence and the inversion silently returns nonsense.

    So order by amplitude, orient the curve so phase broadly increases, then keep
    only the strictly increasing points (first of each repeated/inverted group).

    Returns
    -------
    (phase, amp, n_dropped) or None
        Both arrays ordered by increasing phase. ``None`` if fewer than 2 points
        survive, in which case the reference cannot be inverted.
    """
    order = np.argsort(np.asarray(ref_amps, dtype=float))
    amps = np.asarray(ref_amps, dtype=float)[order]
    phases = np.asarray(ref_phases, dtype=float)[order]

    finite = np.isfinite(amps) & np.isfinite(phases)
    amps, phases = amps[finite], phases[finite]
    if amps.size < 2:
        return None

    # Orient so phase increases with index, then enforce strict monotonicity.
    if phases[-1] < phases[0]:
        amps, phases = amps[::-1], phases[::-1]

    keep = np.ones(phases.size, dtype=bool)
    last = -np.inf
    for i, ph in enumerate(phases):
        if ph > last:
            last = ph
        else:
            keep[i] = False

    n_dropped = int((~keep).sum())
    if keep.sum() < 2:
        return None
    return phases[keep], amps[keep], n_dropped


def _compute_flux_response(
    signal_phase: xr.DataArray,
    ref_cal: Optional[xr.DataArray],
    qubits: list,
    ramsey_flux_amp: float,
    qubit_flux_amp: Optional[float],
) -> Tuple[xr.DataArray, Dict[str, dict]]:
    """Map ``signal_phase(t)`` → Z step response via ``ref_cal``; return branch-risk dict.

    Per qubit: snap each φ to the 2π branch nearest the ref window, interpolate
    the (monotonised) reference to get A_eff, residual
    ``delta = A_eff - ramsey_flux_amp``, step-rise ``y = qubit_flux_amp - delta``.

    Phases that fall outside the reference window are set to NaN rather than
    interpolated: ``np.interp`` clamps to the endpoint amplitude, which would
    silently saturate the step response and bias ``a_dc`` (and therefore every
    tap) instead of dropping the unusable points.
    """
    flux_response = xr.full_like(signal_phase, np.nan, dtype=float)
    branch_risk: Dict[str, dict] = {}
    two_pi = 2 * np.pi

    if ref_cal is None:
        print(
            "WARNING: No reference amplitude sweep found in dataset. "
            "Cannot map phase to flux — flux_response will be NaN."
        )
        return flux_response, branch_risk

    ref_amps = ref_cal.coords["a"].values
    for q in qubits:
        ref_phases = np.asarray(ref_cal.sel(qubit=q.name).values, dtype=float)
        sig_phases = np.asarray(signal_phase.sel(qubit=q.name).values, dtype=float)

        monotonic = _monotonic_reference(ref_amps, ref_phases)
        if monotonic is None:
            print(
                f"WARNING [{q.name}]: reference phase-vs-amplitude curve is not invertible "
                f"(fewer than 2 usable monotonic points); flux_response left as NaN."
            )
            branch_risk[q.name] = {
                "level": "high",
                "code": 2,
                "sig_swing_frac": float("nan"),
                "ref_span_frac": float("nan"),
                "oor_frac": 1.0,
            }
            continue
        ref_ph_s, ref_amp_s, n_dropped = monotonic
        if n_dropped:
            frac = n_dropped / max(len(ref_phases), 1)
            print(
                f"WARNING [{q.name}]: reference phase is non-monotonic in amplitude — dropped "
                f"{n_dropped}/{len(ref_phases)} points ({frac:.0%}) to invert it. A noisy or "
                f"folded reference sweep biases the phase->flux map; consider more shots or a "
                f"narrower ramsey_flux_sweep_range_in_v."
            )

        # Snap each signal phase into the 2pi window centred on the reference.
        ref_center = 0.5 * (ref_ph_s[0] + ref_ph_s[-1])
        adjusted = sig_phases - np.round((sig_phases - ref_center) / two_pi) * two_pi

        # Outside the reference window np.interp would clamp; drop instead.
        out_of_range = (adjusted < ref_ph_s[0]) | (adjusted > ref_ph_s[-1]) | ~np.isfinite(adjusted)
        eff_amp = np.interp(adjusted, ref_ph_s, ref_amp_s)
        eff_amp = np.where(out_of_range, np.nan, eff_amp)
        oor_frac = float(np.mean(out_of_range)) if out_of_range.size else 0.0
        if oor_frac > 0:
            print(
                f"WARNING [{q.name}]: {int(out_of_range.sum())}/{out_of_range.size} delay points "
                f"({oor_frac:.0%}) have a Ramsey phase outside the reference sweep window and were "
                f"dropped. Widen ramsey_flux_sweep_range_in_v (or lower qubit_flux_amplitude_in_v) "
                f"so the reference covers the full phase excursion."
            )

        distortion = eff_amp - ramsey_flux_amp
        flux_response.loc[{"qubit": q.name}] = -distortion + (qubit_flux_amp if qubit_flux_amp is not None else 0)

        # Branch-aliasing diagnostic (shape risk if swing ≳ π…2π).
        ref_span = float(np.ptp(ref_phases)) if ref_phases.size else 0.0
        sig_swing = float(np.ptp(_robust_unwrap_1d(sig_phases))) if sig_phases.size else 0.0
        sig_frac, ref_frac = sig_swing / two_pi, ref_span / two_pi
        if sig_frac >= 1.0:
            level, code = "high", 2
        elif sig_frac >= 0.5:
            level, code = "marginal", 1
        else:
            level, code = "ok", 0
        # Dropped points mean the window was actually exceeded, not just at risk.
        if oor_frac > 0.05 and code < 2:
            level, code = "high", 2
        elif oor_frac > 0 and code < 1:
            level, code = "marginal", 1
        branch_risk[q.name] = {
            "level": level,
            "code": code,
            "sig_swing_frac": sig_frac,
            "ref_span_frac": ref_frac,
            "oor_frac": oor_frac,
        }
        if level != "ok":
            print(
                f"WARNING [{q.name}]: phase->flux branch-aliasing risk = {level.upper()}. "
                f"signal phase swing = {sig_frac:.2f} x 2pi, "
                f"reference span = {ref_frac:.2f} x 2pi, "
                f"points dropped = {oor_frac:.0%}. "
                "Per-point np.round branch selection is exact only while phase stays within "
                "one 2pi window — fitted distortion shape may be aliased "
                "(see warning on flux-response figures)."
            )

    return flux_response, branch_risk


# --- Fit packaging ---


def _extract_relevant_fit_parameters(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the flux step response per qubit and package ``FitParameters``."""
    qubits = node.namespace["qubits"]
    n_exponentials = int(getattr(node.parameters, "n_exponentials", 3))
    t_pulse_ns = float(getattr(node.parameters, "flux_settle_time_in_ns", 0)) or None
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
        fit_results[q.name] = multi_exp_fit_global(
            t_data[mask],
            y_data[mask],
            n_exponentials=n_exponentials,
            t_pulse_ns=t_pulse_ns,
            verbose=True,
            log_callable=node.log,
        )
    return ds, fit_results


# --- Top-level analysis ---


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract phases, map to flux response, and fit exponential cascade.

    Returns ``(ds_fit, fit_results)``: enriched dataset plus per-qubit ``FitParameters``.
    """
    qubits = node.namespace["qubits"]
    qubit_names = [q.name for q in qubits]

    signal_phase, ref_cal = extract_phases(ds)
    flux_response, branch_risk = _compute_flux_response(
        signal_phase,
        ref_cal,
        qubits,
        ramsey_flux_amp=node.parameters.ramsey_flux_amplitude_in_v,
        qubit_flux_amp=getattr(node.parameters, "qubit_flux_amplitude_in_v", None),
    )

    ds = ds.copy()
    ds["signal_phase"] = signal_phase
    if ref_cal is not None:
        ds["ref_phase_cal"] = ref_cal
    ds["flux_response"] = flux_response
    if ref_cal is not None and all(n in branch_risk for n in qubit_names):
        ds["branch_risk_code"] = xr.DataArray(
            [branch_risk[n]["code"] for n in qubit_names],
            dims=["qubit"],
            coords={"qubit": qubit_names},
        )
        ds["branch_sig_swing"] = xr.DataArray(
            [branch_risk[n]["sig_swing_frac"] for n in qubit_names],
            dims=["qubit"],
            coords={"qubit": qubit_names},
            attrs={"long_name": "signal phase peak-to-peak swing", "units": "2*pi"},
        )
        ds["branch_ref_span"] = xr.DataArray(
            [branch_risk[n]["ref_span_frac"] for n in qubit_names],
            dims=["qubit"],
            coords={"qubit": qubit_names},
            attrs={"long_name": "reference phase span", "units": "2*pi"},
        )
        ds["branch_out_of_range"] = xr.DataArray(
            [branch_risk[n]["oor_frac"] for n in qubit_names],
            dims=["qubit"],
            coords={"qubit": qubit_names},
            attrs={"long_name": "delay points outside the reference phase window", "units": "fraction"},
        )

    return _extract_relevant_fit_parameters(ds, node)
