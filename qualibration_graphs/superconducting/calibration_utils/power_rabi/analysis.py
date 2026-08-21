import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits


def _get_operation_name(node: QualibrationNode) -> str:
    """Return the operation name from node parameters (no node name branching)."""
    return getattr(node.parameters, "operation", "EF_x180")


def _fit_oscillation_1d_safe(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Call fit_oscillation; work around qualibration_libs returning float when da has only one dim."""
    if len(da.dims) == 1:
        da = da.expand_dims("_batch")
        return fit_oscillation(da, dim).squeeze("_batch", drop=True)
    return fit_oscillation(da, dim)


def _raw_pi_prefactor(signal_1d: xr.DataArray, freq: float) -> float:
    """Fit-free cross-check of the pi point: amp_prefactor of the FIRST-period extremum of the
    measured Rabi curve. The reported optimum is ``1/(2f)`` (half period, fitted phase discarded);
    this returns where the data actually peaks so the two can be compared. Direction-agnostic: it
    takes the point of the first Rabi period FARTHEST from the zero-amplitude baseline (works whether
    the pi shows up as a max or a min, and avoids confusing the pi extremum with the 2-pi trough).
    NaN on failure."""
    try:
        a = np.asarray(signal_1d["amp_prefactor"].values, dtype=float)
        y = np.asarray(signal_1d.values, dtype=float)
        m = np.isfinite(a) & np.isfinite(y)
        a, y = a[m], y[m]
        if a.size < 3 or not np.isfinite(freq) or freq == 0:
            return float("nan")
        order = np.argsort(a)
        a, y = a[order], y[order]
        # restrict to the first full Rabi period so we pick the first (pi) extremum, not a later one
        win = a <= (a[0] + 1.0 / abs(float(freq)))
        if win.sum() < 3:
            win = np.ones_like(a, dtype=bool)
        aw, yw = a[win], y[win]
        # farthest from the a~0 baseline = the pi excursion (not the 2-pi return-to-baseline trough)
        return float(aw[int(np.nanargmax(np.abs(yw - yw[0])))])
    except Exception:  # pragma: no cover - purely diagnostic, must never break the node
        return float("nan")


@dataclass
class FitParameters:

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    success: bool
    best_readout_frequency: Optional[float] = None  # Set when fitting 2D readout_frequency sweep (best-contrast)
    opt_amp_prefactor_raw: Optional[float] = None  # Fit-free first-period extremum (diagnostic; not written to state)
    # --- v2 fit quality (why success is what it is) ---
    osc_amp_snr: float = float("nan")   # oscillation amplitude / residual scatter (noise fit ~< 3)
    n_periods: float = float("nan")     # Rabi periods visible in the sweep (need >= 0.75)
    pts_per_period: float = float("nan")  # sampling density (need >= 8)
    raw_fit_consistent: bool = True     # fit half-period vs raw first-extremum within 15 %


def _rabi_fit_quality(fit: xr.Dataset, use_state_disc: bool) -> Dict[str, dict]:
    """Post-fit quality of the 1D Rabi oscillation per qubit (pure, never raises).

    The library ``fit_oscillation`` returns *some* sinusoid even on pure noise,
    and the legacy gate (not-NaN + amp<max) let that through — a noise-derived
    pi-amplitude within hardware range was written to state. Grades computed
    here: ``amp_snr`` (fitted amplitude / MAD of the residual — a fit to noise
    scores ~< 3), ``n_periods`` (f x sweep span; below ~0.75 the frequency is
    unconstrained) and ``pts_per_period`` (aliasing guard, need >= ~8).
    Qubits whose signal shape is unavailable (e.g. 2D variants) return NaN and
    are not gated on ``amp_snr``.
    """
    out: Dict[str, dict] = {}
    try:
        from qualibration_libs.analysis.models import oscillation as _osc_model
    except Exception:  # pragma: no cover
        _osc_model = lambda t, a, f, phi, offset: a * np.cos(2 * np.pi * f * t + phi) + offset
    sig_name = "state" if (use_state_disc and "state" in fit) else ("IQ_abs" if "IQ_abs" in fit else "I")
    x = fit.amp_prefactor.values.astype(float) if "amp_prefactor" in fit.coords else None
    for q in [str(v) for v in fit.qubit.values]:
        rec = dict(osc_amp_snr=float("nan"), n_periods=float("nan"),
                   pts_per_period=float("nan"), raw_y_gap=float("nan"))
        try:
            p = fit.fit.sel(qubit=q)
            a = float(p.sel(fit_vals="a")); f = float(p.sel(fit_vals="f"))
            phi = float(p.sel(fit_vals="phi")); off = float(p.sel(fit_vals="offset"))
            if x is not None and np.isfinite(f) and f > 0:
                span = float(x.max() - x.min())
                step = float(np.median(np.abs(np.diff(x)))) or 1.0
                rec["n_periods"] = f * span
                rec["pts_per_period"] = (1.0 / f) / step
                y_da = fit[sig_name].sel(qubit=q)
                if set(y_da.dims) == {"amp_prefactor"}:
                    y = y_da.values.astype(float)
                    resid = y - _osc_model(x, a, f, phi, off)
                    scatter = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-15
                    rec["osc_amp_snr"] = abs(a) / scatter
                # Fit-vs-raw consistency IN SIGNAL SPACE: on a broad flat-topped
                # Rabi hump the raw extremum lands anywhere on the plateau, so an
                # x-distance test misfires; what matters is whether the raw point
                # sits at the same signal level as the fitted pi extremum.
                raw_p = fit.opt_amp_prefactor_raw.sel(qubit=q).values if "opt_amp_prefactor_raw" in fit else np.nan
                fit_p = 1.0 / (2 * f)
                if np.isfinite(raw_p) and abs(a) > 0:
                    y_fit = _osc_model(np.array([fit_p]), a, f, phi, off)[0]
                    y_raw = _osc_model(np.array([float(raw_p)]), a, f, phi, off)[0]
                    rec["raw_y_gap"] = abs(y_raw - y_fit) / (2 * abs(a))
        except Exception:
            pass
        out[q] = rec
    return out


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_amp = (
            f"The calibrated {fit_results[q]['operation']} amplitude: "
            f"{1e3 * fit_results[q]['opt_amp']:.2f} mV "
            f"(x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        )
        # Diagnostic: compare the reported half-period optimum against the raw data peak.
        raw_pref = fit_results[q].get("opt_amp_prefactor_raw")
        fit_pref = fit_results[q]["opt_amp_prefactor"]
        if raw_pref is not None and np.isfinite(raw_pref) and fit_pref:
            s_amp += (
                f"raw-peak prefactor: x{raw_pref:.3f} "
                f"(fit x{fit_pref:.3f}, raw-fit {(raw_pref - fit_pref) / fit_pref * 100:+.1f}%)\n "
            )
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process raw dataset by converting IQ to V and adding amplitude/phase."""
    if not getattr(node.parameters, "use_state_discrimination", False):
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    # Full pulse amplitude (prefactor * operation amplitude) from parameters; no node name branching
    operation = _get_operation_name(node)
    full_amp = np.array(
        [ds.amp_prefactor.values * q.xy.operations[operation].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}

    # Add amplitude and phase (IQ_abs etc.) when using I/Q readout (not state discrimination)
    if hasattr(ds, "I") and not getattr(node.parameters, "use_state_discrimination", False):
        if "readout_frequency" in ds.dims:
            # 2D sweep: add_amplitude_and_phase per readout_frequency slice, then concat
            slices = []
            for freq in ds.readout_frequency.values:
                ds_slice = ds.sel(readout_frequency=freq, drop=True)
                ds_slice = add_amplitude_and_phase(ds_slice, "amp_prefactor", subtract_slope_flag=True)
                slices.append(ds_slice)
            ds = xr.concat(slices, dim=ds.readout_frequency)
        else:
            ds = add_amplitude_and_phase(ds, "amp_prefactor", subtract_slope_flag=True)
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    max_pulses = getattr(node.parameters, "max_number_pulses_per_sweep", 1)
    operation = _get_operation_name(node)
    use_state_disc = getattr(node.parameters, "use_state_discrimination", False)

    if max_pulses == 1 and "readout_frequency" in ds.dims:
        # Best-contrast path: for each qubit, compute contrast at each readout frequency, fit at best
        fit_list = []
        best_freqs = []
        raw_prefs = []
        for q in ds.qubit.values:
            ds_q = ds.sel(qubit=q)
            contrasts = []
            for freq in ds.readout_frequency.values:
                ds_slice = ds_q.sel(readout_frequency=freq)
                if use_state_disc and "state" in ds_slice:
                    fit_vals = _fit_oscillation_1d_safe(ds_slice.state, "amp_prefactor")
                elif "IQ_abs" in ds_slice:
                    fit_vals = _fit_oscillation_1d_safe(ds_slice.IQ_abs, "amp_prefactor")
                else:
                    fit_vals = _fit_oscillation_1d_safe(ds_slice.I, "amp_prefactor")
                a_vals = fit_vals.sel(fit_vals="a").values
                contrasts.append(np.nanmax(np.abs(a_vals)) if np.any(~np.isnan(a_vals)) else np.nan)
            contrasts = np.array(contrasts)
            best_idx = np.nanargmax(contrasts)
            best_freq = float(ds.readout_frequency.values[best_idx])
            best_freqs.append(best_freq)
            # Fit at the best-contrast readout frequency for this qubit
            ds_slice = ds_q.sel(readout_frequency=best_freq)
            if use_state_disc and "state" in ds_slice:
                fit_vals = _fit_oscillation_1d_safe(ds_slice.state, "amp_prefactor")
            elif "IQ_abs" in ds_slice:
                fit_vals = _fit_oscillation_1d_safe(ds_slice.IQ_abs, "amp_prefactor")
            else:
                fit_vals = _fit_oscillation_1d_safe(ds_slice.I, "amp_prefactor")
            fit_list.append(fit_vals.expand_dims(qubit=[q]))
            # Fit-free cross-check of the pi point on the best-contrast signal
            sig = (
                ds_slice.state
                if (use_state_disc and "state" in ds_slice)
                else (ds_slice.IQ_abs if "IQ_abs" in ds_slice else ds_slice.I)
            )
            raw_prefs.append(_raw_pi_prefactor(sig, float(fit_vals.sel(fit_vals="f").values)))
        fit_vals = xr.concat(fit_list, dim="qubit")
        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
        ds_fit = ds_fit.assign(best_readout_frequency=xr.DataArray(best_freqs, coords={"qubit": ds.qubit}))
        ds_fit = ds_fit.assign(opt_amp_prefactor_raw=xr.DataArray(raw_prefs, coords={"qubit": ds.qubit}))
    elif max_pulses == 1:
        # Single-pulse path: drop the (size-1) nb_of_pulses dimension if present, then fit.
        # In 1D mode N_pi_vec == [1], so the dimension exists with length 1 and must be removed,
        # otherwise it survives into the fit and opt_amp_prefactor ends up 1-D (not a scalar).
        ds_fit = ds.isel(nb_of_pulses=0, drop=True) if "nb_of_pulses" in ds.dims else ds
        if use_state_disc:
            fit_var = ds_fit.state
        else:
            fit_var = ds_fit.IQ_abs if "IQ_abs" in ds_fit else ds_fit.I
        fit_vals = _fit_oscillation_1d_safe(fit_var, "amp_prefactor")
        # Fit-free cross-check of the pi point, per qubit
        raw_prefs = [
            _raw_pi_prefactor(
                fit_var.sel(qubit=q) if "qubit" in fit_var.dims else fit_var,
                float(fit_vals.sel(qubit=q, fit_vals="f").values)
                if "qubit" in fit_vals.dims
                else float(fit_vals.sel(fit_vals="f").values),
            )
            for q in fit_var.qubit.values
        ]
        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
        ds_fit = ds_fit.assign(
            opt_amp_prefactor_raw=xr.DataArray(raw_prefs, coords={"qubit": list(fit_var.qubit.values)})
        )
    else:
        # Multi-pulse (error amplification) path: mean over nb_of_pulses, then opt_amp_prefactor from min/max
        ds_fit = ds
        if use_state_disc:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (not ds.nb_of_pulses.data[0] % 2 and operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 and operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    max_pulses = getattr(node.parameters, "max_number_pulses_per_sweep", 1)
    operation = _get_operation_name(node)
    if max_pulses == 1:
        # Pi-pulse prefactor = half-period of the Rabi oscillation = 1/(2f)
        # (phase phi is a readout nuisance parameter, not a physical drive offset)
        factor = 1 / (2 * fit.fit.sel(fit_vals="f"))
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        # Current amplitude from state: EF_x180 for EF node, else node.parameters.operation (e.g. x180)
        if operation == "EF_x180":
            current_amps = xr.DataArray(
                [q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]],
                coords={"qubit": fit.qubit.data},
            )
        else:
            current_amps = xr.DataArray(
                [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
                coords={"qubit": fit.qubit.data},
            )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    else:
        current_amps = xr.DataArray(
            [q.xy.operations[operation].amplitude for q in node.namespace["qubits"]],
            coords={"qubit": fit.qubit.data},
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not (v2: oscillation-quality
    # gates on top of the legacy range checks — a sinusoid fitted to noise can
    # no longer write a pi-amplitude just because it lands inside the hardware
    # range).
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    max_amps = xr.DataArray(
        [lim.max_x180_wf_amplitude for lim in limits],
        coords={"qubit": fit.qubit.data},
    )
    amp_success = fit.opt_amp < max_amps

    use_state_disc = bool(getattr(node.parameters, "use_state_discrimination", False))
    quality = _rabi_fit_quality(fit, use_state_disc) if max_pulses == 1 else {}
    qnames = [str(v) for v in fit.qubit.values]
    snr_arr = np.array([quality.get(q, {}).get("osc_amp_snr", np.nan) for q in qnames])
    nper_arr = np.array([quality.get(q, {}).get("n_periods", np.nan) for q in qnames])
    ppp_arr = np.array([quality.get(q, {}).get("pts_per_period", np.nan) for q in qnames])
    raw_gap_arr = np.array([quality.get(q, {}).get("raw_y_gap", np.nan) for q in qnames])
    # Gates apply only where the metric is defined (NaN -> not gated), so 2D
    # variants and the multi-pulse path keep their legacy behaviour.
    snr_ok = np.isnan(snr_arr) | (snr_arr >= 3.0)
    per_ok = np.isnan(nper_arr) | (nper_arr >= 0.75)
    ppp_ok = np.isnan(ppp_arr) | (ppp_arr >= 8.0)
    raw_ok = np.isnan(raw_gap_arr) | (raw_gap_arr <= 0.2)
    quality_ok = xr.DataArray(snr_ok & per_ok & ppp_ok & raw_ok, coords={"qubit": fit.qubit.data})

    success_criteria = ~nan_success & amp_success & quality_ok
    fit = fit.assign({"success": success_criteria})
    fit = fit.assign({"osc_amp_snr": xr.DataArray(snr_arr, coords={"qubit": fit.qubit.data})})
    fit = fit.assign({"n_periods": xr.DataArray(nper_arr, coords={"qubit": fit.qubit.data})})
    # Populate the FitParameters class with fitted values (incl. best_readout_frequency when from 2D readout sweep)
    best_freq_arr = fit.best_readout_frequency if "best_readout_frequency" in fit else None
    raw_arr = fit.opt_amp_prefactor_raw if "opt_amp_prefactor_raw" in fit else None

    def _raw_for(q):
        if raw_arr is None:
            return None
        v = float(raw_arr.sel(qubit=q).values)
        return v if np.isfinite(v) else None

    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=float(fit.sel(qubit=q).opt_amp_prefactor.values),
            opt_amp=float(fit.sel(qubit=q).opt_amp.values),
            operation=operation,
            success=bool(fit.sel(qubit=q).success.values),
            best_readout_frequency=float(best_freq_arr.sel(qubit=q).values) if best_freq_arr is not None else None,
            opt_amp_prefactor_raw=_raw_for(q),
            osc_amp_snr=float(snr_arr[qnames.index(str(q))]),
            n_periods=float(nper_arr[qnames.index(str(q))]),
            pts_per_period=float(ppp_arr[qnames.index(str(q))]),
            raw_fit_consistent=bool(raw_ok[qnames.index(str(q))]),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
