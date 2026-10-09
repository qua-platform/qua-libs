import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.readout_optimal_integration_weights.weights import (
    ADC_FULL_SCALE,
    apply_iq_imbalance,
    check_weight_limits,
    chunk4,
    envelope,
    estimate_iq_imbalance,
    estimate_single_shot_sigma,
    expand_run_length,
    fold_iq_imbalance_into_weight,
    lowpass_hann,
    normalization_factor,
    omega_t,
    optimal_weight,
    signal_bound,
    snr_for_weight,
)


@dataclass
class OptimalWeightsFit:
    """Stores the relevant optimal-integration-weights fit results for a single qubit."""

    norm: float
    """Factor W was divided by to respect the OPX overflow limits (see weights.normalization_factor).
    May be < 1 -- the weights are then scaled UP to use the available dynamic range."""
    snr: float
    """Predicted SNR of the deployed (possibly variance-weighted) matched-filter weight."""
    snr_constant: float
    """Predicted SNR of today's constant weight (boxcar integration) -- the baseline being improved on."""
    snr_real_envelope: float
    """Predicted SNR of |env_e - env_g| with the best single global phase -- the best a stock,
    non-time-varying-phase ReadoutPulse could achieve."""
    snr_gain: float
    """snr / snr_constant. > 1 means the matched filter beats today's readout."""
    margin_weight: float
    margin_adc_mul: float
    margin_adc_sum: float
    """The three OPX overflow-limit margins (value / limit) on the final, chunked, fixed-point
    -rounded weight. <= 1.0 means within bounds."""
    binding_limit: str
    """Which of the three limits above is closest to binding."""
    adc_headroom: float
    """max(|z_g|, |z_e|) as a fraction of the ADC's +/-0.5 V full scale -- a separate check from
    the weight-overflow margins: if this approaches 1.0, the ADC INPUT itself is near saturation,
    which invalidates the variance estimate the normalization bound relies on."""
    variance_measured: bool
    """Whether the per-sample second moments were streamed (node parameter `measure_variance`).
    When False, `snr`/`snr_constant`/`snr_real_envelope` are computed against a unit pooled
    variance and are therefore in ARBITRARY units (`snr_gain`, a ratio, is unaffected), the
    overflow bound's noise term comes from `sigma_est` instead of a measured variance, and
    `min_snr` is not enforced when deciding `success`."""
    sigma_est: float
    """Per-sample single-shot noise sigma (volts) behind the overflow bound: sqrt of the measured
    pooled variance, or, without one, estimated from the averaged trace's high-frequency residual
    (weights.estimate_single_shot_sigma)."""
    hb_capped: bool
    """Whether the signal bound hit the ADC full scale (+/-0.5 V) somewhere, i.e. the bound is
    the ADC limit rather than mean + SIGMA_NUM*sigma."""
    demod_phase_offset_rad: float
    """Diagnostic only (spec §6): angle between the hardware's own dual_demod output (using
    whatever weights are CURRENTLY active) and what the software demodulation predicts for the
    same weights. A constant, qubit-independent offset here is expected and is left for
    07_iq_blobs to absorb via integration_weights_angle; a value that isn't consistent across
    sub-windows of the trace instead suggests the sign of omega (the IF) is wrong -- see the
    'envelopes' debug plot. NaN if it can't be computed (e.g. a dead trace)."""
    iq_imbalance_abs: float
    iq_imbalance_phase_rad: float
    """The estimated IQ-imbalance ratio `b` (mirror-image model y = x + b*conj(x)), in polar
    form -- estimated from the ground-state trace and folded into the weight (see
    weights.fold_iq_imbalance_into_weight), not applied to the plotted envelopes/traces
    otherwise. Both 0 when no correction was applied (`correct_iq_imbalance=False` or the
    estimate was rejected -- see iq_imbalance_applied/iq_imbalance_reason)."""
    iq_imbalance_applied: bool
    iq_imbalance_reason: str
    """Whether the imbalance correction was folded into the weight, and why not if it wasn't:
    the parameter was off, or one of estimate_iq_imbalance's guards rejected the fit (e.g.
    `|b|` exceeded `max_iq_imbalance`). Empty string when applied."""
    if_configured_hz: float
    if_peak_hz: float
    """Cross-check pair: the configured `resonator.intermediate_frequency` used as the carrier,
    vs. the independent FFT-argmax peak of the ground-state trace (diagnostic only, never used
    as the carrier). A large mismatch usually means a stale `intermediate_frequency` or a wrong
    IF sign -- see log_fitted_results' warning. `if_peak_hz` is NaN when it couldn't be
    computed (mirrors estimate_iq_imbalance's own guards)."""
    success: bool


def has_variance(ds: xr.Dataset) -> bool:
    """Whether the per-sample second moments were acquired (node parameter `measure_variance`),
    i.e. whether a per-sample variance can be computed from this dataset. Checked on the dataset
    rather than on the node parameters so the `load_data_id` replay path is self-describing."""
    return all(f"adcI2_{s}" in ds and f"adcQ2_{s}" in ds for s in ("g", "e"))


def log_fitted_results(ds: xr.Dataset, log_callable=None):
    """Logs the node-specific fitted results for all qubits from the fit xarray Dataset."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    if not has_variance(ds):
        log_callable(
            "measure_variance was off: SNR values below are relative to a unit pooled variance "
            "(arbitrary units) and min_snr is not enforced -- only snr_gain and the overflow "
            "margins decide success. The overflow bound's noise term uses a sigma estimated "
            "from the averaged trace (see sigma_est)."
        )
    for q in ds.qubit.values:
        qd = ds.sel(qubit=q)
        verdict = "SUCCESS!" if bool(qd.success.values) else "FAIL!"
        log_callable(
            f"Optimal weights for qubit {q}: SNR {float(qd.snr):.2f} "
            f"(x{float(qd.snr_gain):.2f} vs constant weights), norm={float(qd.norm):.3g}, "
            f"binding limit '{qd.binding_limit.values!s}' at {float(qd.margin_weight):.2f}/"
            f"{float(qd.margin_adc_mul):.2f}/{float(qd.margin_adc_sum):.2f} "
            f"(weight/adc_mul/adc_sum), single-shot sigma {float(qd.sigma_est) * 1e3:.2f} mV --> {verdict}"
        )
        if bool(qd.hb_capped.values):
            log_callable(
                f"  NOTE for {q}: signal bound reached the ADC full scale ({ADC_FULL_SCALE} V); "
                "the noise is large relative to the ADC range."
            )

        if bool(qd.iq_imbalance_applied.values):
            log_callable(
                f"  IQ imbalance for {q}: |b|={float(qd.iq_imbalance_abs):.3g} "
                f"@ {np.degrees(float(qd.iq_imbalance_phase_rad)):.1f} deg -- corrected."
            )
        else:
            log_callable(f"  IQ imbalance for {q}: not corrected ({qd.iq_imbalance_reason.values!s}).")

        if_configured = float(qd.if_configured_hz)
        if_peak = float(qd.if_peak_hz)
        if np.isfinite(if_peak) and if_configured != 0:
            mismatch = abs(if_peak - if_configured) / abs(if_configured)
            if mismatch > 0.05:
                log_callable(
                    f"  WARNING for {q}: FFT-peak IF ({if_peak / 1e6:.3f} MHz) differs from "
                    f"resonator.intermediate_frequency ({if_configured / 1e6:.3f} MHz) by "
                    f"{100 * mismatch:.0f}% -- check for a stale IF or a wrong IF sign."
                )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Converts raw ADC units to volts (01b's convention: the first moment is divided by 2**12
    once, the second moment by 2**24, i.e. the same factor applied twice -- the sign in 01b's
    `-adc/2**12` squares away for the second moment), computes the per-sample variance for each
    state when the second moments were acquired at all (node parameter `measure_variance` --
    without them only the mean traces are converted), and re-trims every trace to ITS OWN
    qubit's readout-pulse window (dropping the extra
    `smearing` samples QM pads the capture with on each side) -- `readout_length_ns` and
    `trace_window_offset_ns` are per-qubit coords on `ds` (see execute_qua_program), since each
    qubit's readout pulse has its own calibrated length/smearing. The output shares one
    `readout_time` axis sized to the longest qubit, NaN-padded for shorter ones.

    Idempotent: safe to run again on an already-processed dataset (the `load_data_id` replay
    path), since every qubit's own offset there is already 0 and its length unchanged.
    """
    lengths = ds["readout_length_ns"].values.astype(int)
    offsets = ds["trace_window_offset_ns"].values.astype(int)
    width = int(lengths.max())

    with_variance = has_variance(ds)
    for s in ("g", "e"):
        a_i = -ds[f"adcI_{s}"] / 2**12
        a_q = -ds[f"adcQ_{s}"] / 2**12
        converted = {f"a_i_{s}": a_i, f"a_q_{s}": a_q}
        if with_variance:
            var_i = ds[f"adcI2_{s}"] / 2**24 - a_i**2
            var_q = ds[f"adcQ2_{s}"] / 2**24 - a_q**2
            converted[f"var_i_{s}"] = var_i.clip(min=0)
            converted[f"var_q_{s}"] = var_q.clip(min=0)
        ds = ds.assign(converted)

    # Re-trim every readout_time-dimensioned variable (adcI_g/adcQ_g/adcI2_g/adcQ2_g and their
    # a_i_*/a_q_*/var_i_*/var_q_* derivatives, for both states) onto a fresh (qubit, width) NaN
    # array, per qubit's own [offset, offset+length) window -- then swap in the new axis.
    trimmed = {}
    for name, da in ds.data_vars.items():
        if "readout_time" not in da.dims:
            continue
        vals = da.values
        out = np.full((vals.shape[0], width), np.nan, dtype=vals.dtype)
        for i in range(vals.shape[0]):
            length = lengths[i]
            out[i, :length] = vals[i, offsets[i] : offsets[i] + length]
        trimmed[name] = out

    ds = ds.drop_vars([*trimmed.keys(), "readout_time"])
    ds = ds.assign({name: (("qubit", "readout_time"), arr) for name, arr in trimmed.items()})
    ds = ds.assign_coords(
        readout_time=np.arange(width),
        trace_window_offset_ns=("qubit", np.zeros_like(offsets)),
    )
    ds.readout_time.attrs = {"long_name": "time since readout pulse start", "units": "ns"}
    return ds


def _fit_single_qubit(qd: xr.Dataset, q, node: QualibrationNode) -> tuple[dict, OptimalWeightsFit]:
    """Runs the full weights.py pipeline for one qubit, using THIS qubit's own readout length
    (a per-qubit coord on `qd`, since every qubit's readout pulse has its own calibrated length)
    -- every input is sliced down to that qubit's own valid [0, length) window before use, so no
    NaN padding from another qubit's longer trace ever reaches weights.py. Returns (derived
    arrays for plotting, the scalar OptimalWeightsFit). Never raises on a data problem (e.g. a
    dead trace) -- reported as `success=False` with NaN-filled arrays instead, so one bad qubit
    can't abort the whole node. A `length` that isn't a multiple of 4 ns is not guarded here: the
    OPX config (built in create_qua_program/execute_qua_program) already requires every pulse
    length to be a multiple of 4 ns, so the acquisition itself would fail before any qubit ever
    reaches this function -- there's nothing left to check by this point."""
    length = int(qd["readout_length_ns"].values)
    nan_arrays = {
        "env_g": np.full(length, np.nan, dtype=complex),
        "env_e": np.full(length, np.nan, dtype=complex),
        "W": np.full(length, np.nan, dtype=complex),
        "W_norm": np.full(length, np.nan, dtype=complex),
        "W_chunked": np.full(length // 4, np.nan, dtype=complex),
        "hb": np.full(length, np.nan),
    }

    a_i_g, a_q_g = qd["a_i_g"].values[:length], qd["a_q_g"].values[:length]
    a_i_e, a_q_e = qd["a_i_e"].values[:length], qd["a_q_e"].values[:length]
    # Without the second moments (measure_variance=False) there is no noise estimate at all:
    # the overflow bound's noise term is estimated from the averaged trace instead (see
    # estimate_single_shot_sigma below), and the SNRs are computed against a unit pooled
    # variance -- which keeps `snr_gain` and the three-way weight-shape comparison exact, since
    # snr_for_weight scales as 1/sqrt(var) for every shape alike, while making the absolute
    # numbers arbitrary units. See Parameters.measure_variance.
    variance_measured = has_variance(qd)
    if variance_measured:
        var_g = (qd["var_i_g"] + qd["var_q_g"]).values[:length]
        var_e = (qd["var_i_e"] + qd["var_q_e"]).values[:length]
        pooled = (var_g + var_e) / 2
    else:
        var_g = var_e = np.zeros(length)
        pooled = np.ones(length)
    z_g = a_i_g + 1j * a_q_g
    z_e = a_i_e + 1j * a_q_e

    rr = q.resonator
    if_configured_hz = float(rr.intermediate_frequency)
    wt = omega_t(length, rr.intermediate_frequency, rr.time_of_flight, rr.smearing)

    # IQ imbalance: estimated from the RAW ground-state trace (the state unaffected by the qubit
    # drive), then folded into the weight rather than applied to the data -- see
    # weights.fold_iq_imbalance_into_weight for why. Guarded by max_iq_imbalance: a rejected
    # estimate falls back to b=0 (no correction, i.e. today's behaviour), never aborts the qubit.
    if node.parameters.correct_iq_imbalance:
        b, if_peak_hz, iq_reason = estimate_iq_imbalance(
            z_g, rr.intermediate_frequency, max_imbalance=node.parameters.max_iq_imbalance
        )
    else:
        b, if_peak_hz, iq_reason = 0j, float("nan"), "correct_iq_imbalance is False"
    iq_applied = b != 0

    # Raw (uncorrected) envelopes -- what the hardware's own IF demod of the raw, imbalanced ADC
    # actually produces. Needed only for _demod_phase_offset below, which compares against the
    # CURRENTLY deployed weights applied to that same raw-demod baseband.
    env_g_raw = envelope(z_g, wt)
    env_e_raw = envelope(z_e, wt)

    # Single-shot noise for the overflow bound. The hardware limits apply to every shot, not to
    # the averaged trace, so a bound built from the mean alone (the old var=0 behaviour) lets the
    # weights grow until single shots overflow. Measured variance when available, else estimated.
    if variance_measured:
        sigma_est = float(np.sqrt(np.nanmean(pooled)))
    else:
        sigma_est = float(
            np.mean(
                [
                    estimate_single_shot_sigma(env_g_raw, node.parameters.num_shots),
                    estimate_single_shot_sigma(env_e_raw, node.parameters.num_shots),
                ]
            )
        )
        var_g = var_e = np.full(length, sigma_est**2)

    # Envelopes used for the weight fit, the SNR comparison and the diagnostic plots: corrected
    # for IQ imbalance when iq_applied (equivalent to correcting the raw trace first -- see
    # weights.apply_iq_imbalance -- then demodulating; a no-op when b==0, i.e. identical to
    # env_g_raw/env_e_raw above).
    env_g = envelope(apply_iq_imbalance(z_g, b), wt)
    env_e = envelope(apply_iq_imbalance(z_e, b), wt)

    # Optional smoothing (default off): a zero-phase low-pass, well above any real resonator
    # bandwidth, to suppress averaging jitter and any residual carrier leakage (plot_weight_spectrum)
    # at the source. Deliberately NOT applied to env_g_raw/env_e_raw (_demod_phase_offset needs
    # the unfiltered hardware-comparable baseband) or to z_g/z_e (the overflow bound below needs
    # the raw trace amplitude the hardware actually sees) -- see Parameters.smooth_bandwidth_hz.
    if node.parameters.smooth_bandwidth_hz is not None:
        env_g = lowpass_hann(env_g, node.parameters.smooth_bandwidth_hz)
        env_e = lowpass_hann(env_e, node.parameters.smooth_bandwidth_hz)

    use_var = node.parameters.use_variance_weighting
    W_x = optimal_weight(env_g, env_e, var_g if use_var else None, var_e if use_var else None)
    # Fold the imbalance correction into the weight (a no-op when b==0) so the SINGLE complex
    # weight written to hardware -- applied to the element's own IF-demod of the RAW, uncorrected
    # ADC (env_g_raw/env_e_raw above) -- reproduces the corrected matched filter. Everything
    # downstream uses this W.
    W = fold_iq_imbalance_into_weight(W_x, b, wt)
    # Overflow normalization must see the RAW trace amplitudes: that's what the hardware
    # actually multiplies the weight against.
    norm = normalization_factor(W, [z_g, z_e], [var_g, var_e])

    adc_headroom = float(np.max([np.abs(z_g).max(), np.abs(z_e).max()]) / 0.5)

    if not (np.isfinite(norm) and norm > 0):
        fit = OptimalWeightsFit(
            norm=float(norm) if np.isfinite(norm) else float("nan"),
            snr=0.0,
            snr_constant=0.0,
            snr_real_envelope=0.0,
            snr_gain=0.0,
            margin_weight=float("nan"),
            margin_adc_mul=float("nan"),
            margin_adc_sum=float("nan"),
            binding_limit="none",
            adc_headroom=adc_headroom,
            variance_measured=variance_measured,
            sigma_est=sigma_est,
            hb_capped=False,
            demod_phase_offset_rad=float("nan"),
            iq_imbalance_abs=float(np.abs(b)),
            iq_imbalance_phase_rad=float(np.angle(b)),
            iq_imbalance_applied=iq_applied,
            iq_imbalance_reason=iq_reason,
            if_configured_hz=if_configured_hz,
            if_peak_hz=if_peak_hz,
            success=False,
        )
        return {**nan_arrays, "env_g": env_g, "env_e": env_e}, fit

    W_norm = W / norm
    W_chunked = chunk4(W_norm)
    hb = signal_bound((z_g, z_e), (var_g, var_e))
    hb_capped = bool(np.any(hb >= ADC_FULL_SCALE))
    hb_chunked = chunk4(hb)
    limits = check_weight_limits(W_chunked, hb_chunked)

    # D is the (IQ-imbalance-corrected) envelope difference the filter is matched to.
    # snr_for_weight's magnitude form is a global-phase-invariant upper bound on the Re[...] the
    # hardware actually produces (07_iq_blobs realizes that phase by fitting
    # integration_weights_angle), so comparing the folded W against this D is a reasonable proxy
    # for the deployed filter's SNR even though the fold is exact only for the Re[...] output.
    D = env_e - env_g
    snr_const = snr_for_weight(np.ones_like(D), D, pooled)
    snr_real_env = snr_for_weight(np.abs(D), D, pooled)
    snr_full = snr_for_weight(W, D, pooled)
    snr_gain = snr_full / snr_const if snr_const > 0 else float("inf")

    # Diagnostic only: computed just for debug_plots runs, NaN otherwise.
    demod_phase_offset_rad = (
        _demod_phase_offset(qd, q, env_g_raw, env_e_raw, length) if node.parameters.debug_plots else float("nan")
    )

    margins_ok = limits["weight"] <= 1.0 and limits["adc_mul"] <= 1.0 and limits["adc_sum"] <= 1.0
    # min_snr is an absolute threshold, so it only means anything when the SNRs were computed
    # against a measured variance; without one, snr_gain carries the whole verdict.
    snr_ok = snr_full >= node.parameters.min_snr if variance_measured else True
    success = bool(margins_ok and snr_ok and snr_gain >= 1.0)

    fit = OptimalWeightsFit(
        norm=float(norm),
        snr=float(snr_full),
        snr_constant=float(snr_const),
        snr_real_envelope=float(snr_real_env),
        snr_gain=float(snr_gain),
        margin_weight=limits["weight"],
        margin_adc_mul=limits["adc_mul"],
        margin_adc_sum=limits["adc_sum"],
        binding_limit=limits["binding_limit"],
        adc_headroom=adc_headroom,
        variance_measured=variance_measured,
        sigma_est=sigma_est,
        hb_capped=hb_capped,
        demod_phase_offset_rad=demod_phase_offset_rad,
        iq_imbalance_abs=float(np.abs(b)),
        iq_imbalance_phase_rad=float(np.angle(b)),
        iq_imbalance_applied=iq_applied,
        iq_imbalance_reason=iq_reason,
        if_configured_hz=if_configured_hz,
        if_peak_hz=if_peak_hz,
        success=success,
    )
    arrays = {
        "env_g": env_g,
        "env_e": env_e,
        "W": W,
        "W_norm": W_norm,
        "W_chunked": W_chunked,
        "hb": hb,
    }
    return arrays, fit


def _demod_phase_offset(qd: xr.Dataset, q, env_g: np.ndarray, env_e: np.ndarray, length: int) -> float:
    """Diagnostic (spec §6): compares the hardware's own dual_demod output (I_g/Q_g/I_e/Q_e,
    obtained for free from the same measure() calls that streamed the ADC trace, using whatever
    weights are CURRENTLY active on the resonator) against what software demodulation predicts
    for those same weights. Returns NaN if the current weights or the predicted difference can't
    be evaluated (e.g. a dead trace)."""
    try:
        pulse = q.resonator.operations["readout"]
        current = pulse.integration_weights_function()
        w_real = expand_run_length(current["real"])[:length]
        w_imag = expand_run_length(current["imag"])[:length]
        if len(w_real) < length or len(w_imag) < length:
            return float("nan")
        W_current = w_real + 1j * w_imag

        predicted_diff = np.sum(np.conj(W_current) * (env_e - env_g))
        if np.abs(predicted_diff) == 0 or not np.isfinite(predicted_diff):
            return float("nan")

        # Dual_demod's I/Q convention (matched by W's own real/imag -> iw1/iw2/iw3 mapping,
        # see the module docstring) is I = Re[conj(W)*env], Q = -Im[conj(W)*env] -- i.e.
        # conj(W)*env = I - i*Q, NOT I + i*Q. Reconstructing predicted_diff's own convention
        # from the hardware's I/Q therefore needs a MINUS sign on the Q part.
        hw_diff = complex(
            float(qd["I_e"]) - float(qd["I_g"]),
            -(float(qd["Q_e"]) - float(qd["Q_g"])),
        )
        return float(np.angle(hw_diff / predicted_diff))
    except Exception:  # noqa: BLE001 -- diagnostic only, must not abort the node
        return float("nan")


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, dict[str, OptimalWeightsFit]]:
    """Runs the matched-filter construction (weights.py) for every qubit and assembles the
    results into a fit dataset: the raw/processed data plus per-qubit derived arrays
    (env_g/env_e/W/W_norm/W_chunked/hb) and scalar coords (norm, snr*, margins, success,
    iq_imbalance_*, if_*, ...). Each qubit's arrays are computed at its own (possibly shorter)
    native length and then NaN-padded onto the shared `readout_time`/`chunk_time` axes sized to
    the longest qubit.
    """
    qubits = node.namespace["qubits"]
    width = ds.sizes["readout_time"]
    chunk_width = width // 4

    per_qubit_arrays = {}
    fit_results: dict[str, OptimalWeightsFit] = {}
    for q in qubits:
        qd = ds.sel(qubit=q.name)
        arrays, fit = _fit_single_qubit(qd, q, node)
        per_qubit_arrays[q.name] = arrays
        fit_results[q.name] = fit

    names = [q.name for q in qubits]

    def pad_stack(key, out_width, dtype):
        out = np.full((len(names), out_width), np.nan, dtype=dtype)
        for i, name in enumerate(names):
            arr = per_qubit_arrays[name][key]
            out[i, : len(arr)] = arr
        return out

    ds_fit = ds.assign(
        {
            "env_g": (("qubit", "readout_time"), pad_stack("env_g", width, complex)),
            "env_e": (("qubit", "readout_time"), pad_stack("env_e", width, complex)),
            "W": (("qubit", "readout_time"), pad_stack("W", width, complex)),
            "W_norm": (("qubit", "readout_time"), pad_stack("W_norm", width, complex)),
            "hb": (("qubit", "readout_time"), pad_stack("hb", width, float)),
            "W_chunked": (("qubit", "chunk_time"), pad_stack("W_chunked", chunk_width, complex)),
        }
    )
    ds_fit = ds_fit.assign_coords(chunk_time=("chunk_time", 4 * np.arange(chunk_width)))
    ds_fit.chunk_time.attrs = {"long_name": "time since readout pulse start", "units": "ns"}

    for field in (
        "norm",
        "snr",
        "snr_constant",
        "snr_real_envelope",
        "snr_gain",
        "margin_weight",
        "margin_adc_mul",
        "margin_adc_sum",
        "adc_headroom",
        "sigma_est",
        "demod_phase_offset_rad",
        "iq_imbalance_abs",
        "iq_imbalance_phase_rad",
        "if_configured_hz",
        "if_peak_hz",
    ):
        ds_fit = ds_fit.assign_coords({field: ("qubit", [getattr(fit_results[name], field) for name in names])})
    ds_fit = ds_fit.assign_coords(
        binding_limit=("qubit", [fit_results[name].binding_limit for name in names]),
        success=("qubit", [fit_results[name].success for name in names]),
        hb_capped=("qubit", [fit_results[name].hb_capped for name in names]),
        iq_imbalance_applied=("qubit", [fit_results[name].iq_imbalance_applied for name in names]),
        iq_imbalance_reason=("qubit", [fit_results[name].iq_imbalance_reason for name in names]),
    )

    return ds_fit, fit_results
