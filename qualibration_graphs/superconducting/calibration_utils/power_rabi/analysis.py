import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits

# Fit-quality gates applied to the single-pulse Rabi fit (see _rabi_fit_quality): the library's
# fit_oscillation returns *some* sinusoid even on pure noise, and a noise-derived pi-amplitude that
# happens to land inside the hardware range must not be reported as a success.
MIN_OSC_AMP_SNR = 3.0  # fitted amplitude / residual scatter
MIN_N_PERIODS = 0.75  # Rabi periods spanned by the sweep; below this the frequency is unconstrained
MIN_PTS_PER_PERIOD = 8.0  # sweep points per Rabi period (aliasing guard)
MAX_RAW_FIT_GAP = 0.2  # max allowed gap, in signal space, between the fit and the raw-peak cross-check


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
    """Stores the relevant power Rabi fit parameters for a single qubit."""

    opt_amp_prefactor: float
    """Amplitude prefactor, relative to the operation's nominal amplitude, that yields a pi pulse."""

    opt_amp: float
    """Absolute pulse amplitude, in volts, that yields a pi pulse."""

    operation: str
    """Name of the calibrated operation, e.g. "x180" or "EF_x180"."""

    success: bool
    """Whether the fit passed both the amplitude-range check and the oscillation-quality gates."""

    opt_amp_prefactor_raw: float | None = None
    """Fit-free amplitude prefactor of the first-period extremum; diagnostic only, not written to state."""

    osc_amp_snr: float = float("nan")
    """Fitted oscillation amplitude divided by residual scatter; NaN when not computed."""

    n_periods: float = float("nan")
    """Number of Rabi periods spanned by the amplitude sweep; NaN when not computed."""

    pts_per_period: float = float("nan")
    """Number of sweep points sampled per Rabi period; NaN when not computed."""

    raw_fit_consistent: bool = True
    """Whether the fit-free raw peak agrees with the fitted pi point in signal space."""


def _rabi_fit_quality(fit: xr.Dataset, use_state_disc: bool) -> dict[str, dict[str, float]]:
    """Post-fit quality of the 1D Rabi oscillation per qubit (pure, never raises).

    Grades computed here: ``osc_amp_snr`` (fitted amplitude / MAD of the residual — a fit to noise
    scores below ``MIN_OSC_AMP_SNR``), ``n_periods`` (f x sweep span; below ``MIN_N_PERIODS`` the
    frequency is unconstrained), ``pts_per_period`` (aliasing guard) and ``raw_y_gap`` (fit-vs-raw
    consistency in signal space, since on a broad flat-topped Rabi hump the raw extremum's x-position
    alone is not a reliable check). Qubits whose signal shape is unavailable return NaN for the
    corresponding metric, which is treated as "not gated" by the caller.
    """
    out: dict[str, dict[str, float]] = {}
    sig_name = "state" if (use_state_disc and "state" in fit) else ("IQ_abs" if "IQ_abs" in fit else "I")
    x = fit.amp_prefactor.values.astype(float) if "amp_prefactor" in fit.coords else None
    for q in [str(v) for v in fit.qubit.values]:
        rec: dict[str, float] = dict(
            osc_amp_snr=float("nan"), n_periods=float("nan"), pts_per_period=float("nan"), raw_y_gap=float("nan")
        )
        try:
            p = fit.fit.sel(qubit=q)
            a, f = float(p.sel(fit_vals="a")), float(p.sel(fit_vals="f"))
            phi, off = float(p.sel(fit_vals="phi")), float(p.sel(fit_vals="offset"))
            if x is not None and np.isfinite(f) and f > 0:
                span = float(x.max() - x.min())
                step = float(np.median(np.abs(np.diff(x)))) or 1.0
                rec["n_periods"] = f * span
                rec["pts_per_period"] = (1.0 / f) / step
                y_da = fit[sig_name].sel(qubit=q)
                if set(y_da.dims) == {"amp_prefactor"}:
                    resid = y_da.values.astype(float) - oscillation(x, a, f, phi, off)
                    scatter = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-15
                    rec["osc_amp_snr"] = abs(a) / scatter
                raw_p = fit.opt_amp_prefactor_raw.sel(qubit=q).values if "opt_amp_prefactor_raw" in fit else np.nan
                fit_p = 1.0 / (2 * f)
                if np.isfinite(raw_p) and abs(a) > 0:
                    y_fit = oscillation(np.array([fit_p]), a, f, phi, off)[0]
                    y_raw = oscillation(np.array([float(raw_p)]), a, f, phi, off)[0]
                    rec["raw_y_gap"] = abs(y_raw - y_fit) / (2 * abs(a))
        except Exception:  # pragma: no cover - purely diagnostic, must never break the node
            pass
        out[q] = rec
    return out


def _gate_ok(value: float, threshold: float, minimum: bool) -> bool:
    """True if `value` clears `threshold`; a NaN value (metric not computed) is never gated."""
    return np.isnan(value) or (value >= threshold if minimum else value <= threshold)


def _raw_gap_ok(rec: dict[str, float]) -> bool:
    """True if the fit-free raw peak agrees with the fitted pi point in signal space (or wasn't computed)."""
    return _gate_ok(rec.get("raw_y_gap", float("nan")), MAX_RAW_FIT_GAP, minimum=False)


def _passes_quality_gates(rec: dict[str, float]) -> bool:
    """True if every fit-quality metric present in `rec` clears its gate; a missing (NaN) metric is not gated."""
    return (
        _gate_ok(rec.get("osc_amp_snr", float("nan")), MIN_OSC_AMP_SNR, minimum=True)
        and _gate_ok(rec.get("n_periods", float("nan")), MIN_N_PERIODS, minimum=True)
        and _gate_ok(rec.get("pts_per_period", float("nan")), MIN_PTS_PER_PERIOD, minimum=True)
        and _raw_gap_ok(rec)
    )


def log_fitted_results(fit_results: dict[str, dict], log_callable: Callable[[str], None] | None = None) -> None:
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : Callable[[str], None], optional
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


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Process raw dataset by converting IQ to V and adding amplitude/phase."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    operation = node.parameters.operation
    full_amp = np.array(
        [ds.amp_prefactor.values * q.xy.operations[operation].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}

    if hasattr(ds, "I") and not node.parameters.use_state_discrimination:
        ds = add_amplitude_and_phase(ds, "amp_prefactor", subtract_slope_flag=True)
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, dict[str, FitParameters]]:
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
    operation = node.parameters.operation
    use_state_disc = node.parameters.use_state_discrimination

    if max_pulses == 1:
        # Single-pulse path: drop the (size-1) nb_of_pulses dimension if present, then fit.
        # In 1D mode N_pi_vec == [1], so the dimension exists with length 1 and must be removed,
        # otherwise it survives into the fit and opt_amp_prefactor ends up 1-D (not a scalar).
        ds_fit = ds.isel(nb_of_pulses=0, drop=True) if "nb_of_pulses" in ds.dims else ds
        if use_state_disc:
            fit_var = ds_fit.state
        else:
            fit_var = ds_fit.IQ_abs if "IQ_abs" in ds_fit else ds_fit.I
        fit_vals = fit_oscillation(fit_var, "amp_prefactor")
        # Fit-free cross-check of the pi point, per qubit
        raw_prefs = [
            _raw_pi_prefactor(fit_var.sel(qubit=q), float(fit_vals.sel(qubit=q, fit_vals="f").values))
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


def _extract_relevant_fit_parameters(
    fit: xr.Dataset, node: QualibrationNode
) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    max_pulses = getattr(node.parameters, "max_number_pulses_per_sweep", 1)
    operation = node.parameters.operation
    current_amps = xr.DataArray(
        [q.xy.operations[operation].amplitude for q in node.namespace["qubits"]],
        coords={"qubit": fit.qubit.data},
    )
    if max_pulses == 1:
        # Pi-pulse prefactor = half-period of the Rabi oscillation = 1/(2f)
        # (phase phi is a readout nuisance parameter, not a physical drive offset)
        factor = 1 / (2 * fit.fit.sel(fit_vals="f"))
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        fit = fit.assign({"opt_amp": factor * current_amps})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}
    else:
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not: oscillation-quality gates on top of the
    # legacy range check, so a sinusoid fitted to noise can no longer pass just because it lands
    # inside the hardware range.
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    max_amps = xr.DataArray(
        [lim.max_x180_wf_amplitude for lim in limits],
        coords={"qubit": fit.qubit.data},
    )
    amp_success = fit.opt_amp < max_amps

    quality = _rabi_fit_quality(fit, node.parameters.use_state_discrimination) if max_pulses == 1 else {}
    quality_ok = xr.DataArray(
        [_passes_quality_gates(quality.get(str(q), {})) for q in fit.qubit.values],
        coords={"qubit": fit.qubit.data},
    )

    success_criteria = ~nan_success & amp_success & quality_ok
    fit = fit.assign({"success": success_criteria})

    raw_arr = fit.opt_amp_prefactor_raw if "opt_amp_prefactor_raw" in fit else None

    def _raw_for(q: str) -> float | None:
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
            opt_amp_prefactor_raw=_raw_for(q),
            osc_amp_snr=quality.get(str(q), {}).get("osc_amp_snr", float("nan")),
            n_periods=quality.get(str(q), {}).get("n_periods", float("nan")),
            pts_per_period=quality.get(str(q), {}).get("pts_per_period", float("nan")),
            raw_fit_consistent=_raw_gap_ok(quality.get(str(q), {})),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
