"""Pure-numpy (plus scipy.signal for the optional smoothing filter) core for the
optimal-integration-weights node (spec §§2-5, 9): software demodulation, matched-filter weight
construction, OPX fixed-point overflow limits, and the predicted-SNR comparisons the node
reports. No QUA/xarray dependency here on purpose -- this module is unit-testable standalone and
has no hardware/config side effects.

Overflow limits per the OPX demod guide (qm-docs.qualang.io/guides/demod), used at 25% of
each hardware ceiling:
    MAX_WEIGHT             -- |w[n]| for every weight array entry.
    MAX_WEIGHT_ADC_MUL     -- |w[n] * adc[n]| for any single sample.
    MAX_WEIGHT_ADC_MUL_SUM -- |sum_n w[n] * adc[n]| across the whole integration window.
"""

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.signal import filtfilt, firwin

MAX_WEIGHT = 1024 - 2**-15
MAX_WEIGHT_ADC_MUL = 2 - 2**-19
MAX_WEIGHT_ADC_MUL_SUM = 32768 - 2**-16
TOLERANCE = 0.25
SIGMA_NUM = 5
"""Single-shot noise multiplier in the signal bound. The overflow limits apply to EVERY sample of
EVERY shot, and a shot has ~1000 samples: at 2 sigma ~2% of samples (i.e. nearly every shot)
exceed the bound; 5 sigma makes an exceedance vanishingly rare."""
ADC_FULL_SCALE = 0.5
"""ADC input range in volts (+/-0.5 V): no sample can exceed it, so it caps the signal bound."""
SIGMA_ESTIMATE_BANDWIDTH_HZ = 20e6
WEIGHT_FIXED_POINT_ACCURACY = 2**-15
MAX_IQ_IMBALANCE = 0.5
"""Reject-and-fall-back threshold on |b| (the y = x + b*conj(x) mirror-image ratio). Physical
mixer/ADC imbalance is a few percent; anything approaching 1 is a failed fit (e.g. from a
DC-dominated spectrum or a too-short trace), not a real imbalance -- estimate_iq_imbalance falls
back to b=0 (no correction) rather than deploy a value that large."""


def omega_t(length_ns: int, f_if_hz: float, tof_ns: int, smearing_ns: int) -> np.ndarray:
    """Carrier phase of each 1 ns ADC sample (spec §2). `f_if_hz` is the readout element's
    intermediate frequency; `tof_ns`/`smearing_ns` are its time_of_flight/smearing.

    The additive time offset (tof - smearing/2) only sets a *constant* phase across the whole
    trace -- it rotates the g/e separation in the IQ plane but doesn't change the shape of the
    weight, and is left for `07_iq_blobs` to absorb via `integration_weights_angle`, same as
    for every other readout on this chip. Only the magnitude and SIGN of `f_if_hz` change the
    shape of omega_t (a wrong sign shows up as fast residual oscillation in the demodulated
    envelope -- see the node's `envelopes` debug plot).
    """
    ts = np.arange(length_ns) + (tof_ns - round(smearing_ns / 2))
    return 2 * np.pi * f_if_hz * 1e-9 * ts


def envelope(z: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Demodulate a complex MW-FEM trace to baseband: env(t) = z(t) * exp(-i*omega*t).

    No low-pass filter, no x2 scaling (spec §2): those exist only to remove the 2*omega image
    term produced by demodulating a *real* signal, which doesn't arise here -- the MW-FEM trace
    is already the complex IQ pair.
    """
    return z * np.exp(-1j * wt)


def lowpass_hann(sig: np.ndarray, bandwidth_hz: float, fs: float = 1e9) -> np.ndarray:
    """Zero-phase Hann-windowed FIR low-pass, applied independently to the real and imaginary
    parts (so it commutes with conjugation, subtraction and the IQ-imbalance fold -- all linear
    in the real/imaginary components -- meaning it makes no difference whether this is called on
    `env_g`/`env_e` individually or on `W` directly).

    Meant to run on a demodulated (baseband) envelope, not a still-oscillating raw trace: a
    low-pass centered at DC only means "keep the slow, physical part of the signal" once the IF
    carrier has already been removed by `envelope()`.

    Zero-phase (`scipy.signal.filtfilt`, not a causal single-pass filter) matters specifically
    for a matched filter: any group delay would time-shift the weight relative to the pulse,
    which costs real SNR, not just cosmetics. `padtype="even"` (mirror the edge samples) avoids
    both the periodic-wraparound (Gibbs) artifact a naive FFT-multiply low-pass would introduce
    on this transient, non-periodic trace, and the jump `padtype="odd"` (filtfilt's own default)
    would introduce at a tail that isn't already close to zero -- the field is near zero at
    t=0 (before ring-up) but not necessarily zero at the tail (right before depletion).

    `numtaps` (the FIR filter length) is derived from the trace length rather than exposed as a
    parameter: `filtfilt` requires `padlen = 3*numtaps < len(sig) - 1` for its default edge
    padding, so it is clamped to fit even a short readout pulse (with margin, and rounded down to
    odd). Returns `sig` unfiltered (no error) when the trace is too short for any reasonable
    filter -- consistent with this module's other guards (e.g. `estimate_iq_imbalance`'s) -- one
    qubit's pulse being too short to smooth shouldn't abort a multi-qubit batch.
    """
    n = len(sig)
    max_numtaps = (n - 2) // 3
    if max_numtaps % 2 == 0:
        max_numtaps -= 1
    numtaps = min(201, max_numtaps)
    if numtaps < 5:
        return sig
    taps = firwin(numtaps, cutoff=bandwidth_hz, window="hann", fs=fs)
    return filtfilt(taps, [1.0], sig.real, padtype="even") + 1j * filtfilt(taps, [1.0], sig.imag, padtype="even")


def estimate_iq_imbalance(
    z: np.ndarray,
    f_if_hz: float,
    fs: float = 1e9,
    n_fft: int = 2**15,
    max_imbalance: float = MAX_IQ_IMBALANCE,
) -> tuple[complex, float, str]:
    """Estimate the IQ-imbalance ratio `b` in the mirror-image model `y[n] = x[n] + b*conj(x[n])`
    from a single raw (non-demodulated) complex ADC trace `z`, plus the FFT-peak IF as an
    independent cross-check on `f_if_hz` (diagnostic only -- never used as the carrier; a
    mismatch usually means a stale `resonator.intermediate_frequency` or a wrong IF sign).

    The signal bin is located from the CONFIGURED `f_if_hz`, not from an FFT argmax -- unlike an
    argmax, this can't silently lock onto the mirror image and flip the correction's sign.

    Returns `(b, if_peak_hz, reason)`. `reason` is `""` on success; otherwise `b` is forced to
    `0` (no correction) and `reason` explains why: the trace was unusable, or `|b|` exceeded
    `max_imbalance` (see that constant's docstring) -- a value that large indicates a failed fit,
    not a real imbalance, so falling back to no correction is safer than deploying it.
    """
    z = np.asarray(z)
    z = z[np.isfinite(z)]
    # A handful of IF periods are needed for the mirror bin to carry a resolvable signal.
    min_len = max(8, int(4 * fs / abs(f_if_hz))) if f_if_hz else None
    if f_if_hz == 0 or min_len is None or z.size < min_len:
        return 0j, float("nan"), "IF is zero or trace too short to resolve a mirror bin"

    z = z - np.mean(z)  # DC removal for the FFT only -- see module docstring
    n = z.size
    n_fft = max(n_fft, n)
    z_f = np.fft.fft(z, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1 / fs)

    bin_signal = round(f_if_hz * n_fft / fs)
    if bin_signal % n_fft == 0:
        return 0j, float("nan"), "signal bin coincides with DC/mirror bin"

    peak_bin = int(np.argmax(np.abs(z_f)))
    if_peak_hz = float(freqs[peak_bin])

    denom = z_f[bin_signal % n_fft].conj()
    if denom == 0 or not np.isfinite(denom):
        return 0j, if_peak_hz, "zero/non-finite amplitude at the signal bin"

    b = z_f[(-bin_signal) % n_fft] / denom
    if not np.isfinite(b):
        return 0j, if_peak_hz, "non-finite imbalance estimate"
    if np.abs(b) > max_imbalance:
        return 0j, if_peak_hz, f"|b|={np.abs(b):.3g} exceeds max_iq_imbalance={max_imbalance:.3g}"
    return complex(b), if_peak_hz, ""


def apply_iq_imbalance(z: np.ndarray, b: complex) -> np.ndarray:
    """Invert the mirror-image model `y = x + b*conj(x)`: recover `x` from a raw trace `y`.
    Returns `y` unchanged when `b == 0` (no correction estimated/applied)."""
    if b == 0:
        return z
    denom = 1 - np.abs(b) ** 2
    return (z - b * np.conj(z)) / denom


def fold_iq_imbalance_into_weight(W: np.ndarray, b: complex, wt: np.ndarray) -> np.ndarray:
    """Fold an IQ-imbalance correction into the weight instead of the data.

    The weight (the "cosine"/"sine" arrays QuAM writes into the element config) is applied on
    hardware to the element's own IF-demodulated baseband of the RAW, uncorrected ADC -- i.e. to
    env_raw = envelope(z_raw, wt), the same quantity `envelope()` above computes in software, NOT
    to z_raw directly (the element's `intermediate_frequency` handles that down-conversion by
    itself; the weight only windows/matches the result). So correcting the envelope in software
    analysis alone would produce a weight that is only valid for data the hardware never
    actually produces. Deriving `Re[sum conj(W_x)*env_corr] = Re[sum conj(W_eff)*env_raw]` from
    `apply_iq_imbalance`'s inverse (in envelope domain, `env_corr = (env_raw -
    b*exp(-2i*wt)*conj(env_raw)) / (1-|b|^2)`) gives the same mirror-image form back for the
    weight:

        W_eff[n] = (W_x[n] - b*exp(-2i*wt[n])*conj(W_x[n])) / (1 - |b|^2)

    so a single complex weight applied to the raw trace reproduces the corrected filter's I
    output exactly (Q is no longer its orthogonal partner when b != 0 -- acceptable, since state
    discrimination thresholds on I and 07_iq_blobs re-fits integration_weights_angle/threshold
    on I afterward). Returns `W_x` unchanged when `b == 0`.
    """
    if b == 0:
        return W
    denom = 1 - np.abs(b) ** 2
    return (W - b * np.exp(-2j * wt) * np.conj(W)) / denom


def optimal_weight(
    env_g: np.ndarray,
    env_e: np.ndarray,
    var_g: np.ndarray | None = None,
    var_e: np.ndarray | None = None,
    variance_floor_percentile: float = 10.0,
) -> np.ndarray:
    """Matched-filter weight W(t) = env_e(t) - env_g(t).

    Repo convention: |e> lands at higher I (the spec's `env_g - env_e` is the opposite sign,
    which would invert every existing readout threshold/state-discrimination convention here).

    If both variances are given, divides by the pooled per-sample variance (spec §3's optional
    noise-weighting switch), floored at a low percentile of the positive pooled variance so a
    handful of near-zero-variance samples (e.g. right at pulse turn-on, before any signal has
    arrived) can't blow the weight up to a spurious large value.
    """
    W = env_e - env_g
    if var_g is not None and var_e is not None:
        pooled = (np.asarray(var_g) + np.asarray(var_e)) / 2
        positive = pooled[pooled > 0]
        floor = np.percentile(positive, variance_floor_percentile) if positive.size else 1.0
        W = W / np.maximum(pooled, floor)
    return W


def estimate_single_shot_sigma(
    env: np.ndarray,
    num_shots: int,
    bandwidth_hz: float = SIGMA_ESTIMATE_BANDWIDTH_HZ,
    fs: float = 1e9,
) -> float:
    """Estimate the per-sample SINGLE-SHOT noise sigma (sqrt of var_i + var_q, the same total
    variance spec §1 defines) from an averaged, demodulated envelope alone.

    The streamed second moment is unusable on ADC streams (see Parameters.measure_variance), and
    without any noise term the overflow bound sees only the deterministic mean trace, which is
    typically far below the single-shot amplitude -- the weights then come out far too large.
    The averaged trace still carries the noise: its high-frequency residual around a smooth
    (low-passed) copy has variance sigma^2 / num_shots per sample, assuming ~white noise at the
    1 ns sampling. The power the low-pass removed (fraction 2*bandwidth/fs) is added back. Real
    signal faster than `bandwidth_hz` (ring-up edges) inflates the residual, which errs on the
    safe (larger-sigma) side. Returns 0.0 when the trace is too short to filter.
    """
    env = np.asarray(env)
    env = env[np.isfinite(env)]
    smooth = lowpass_hann(env, bandwidth_hz, fs=fs)
    if smooth is env:  # too short to filter
        return 0.0
    edge = min(100, env.size // 4)  # filter edge transients
    resid = (env - smooth)[edge : env.size - edge]
    if resid.size == 0:
        return 0.0
    kept_fraction = 1 - 2 * bandwidth_hz / fs
    return float(np.sqrt(num_shots * np.mean(np.abs(resid) ** 2) / kept_fraction))


def signal_bound(
    z_list: Sequence[np.ndarray],
    var_list: Sequence[np.ndarray],
    sigma_num: float = SIGMA_NUM,
) -> np.ndarray:
    """Per-sample upper bound on the single-shot ADC amplitude the weights multiply:
    max over states of |mean| + sigma_num*sigma, capped at the ADC full scale."""
    hb = np.max(
        [np.abs(z) + sigma_num * np.sqrt(np.maximum(var, 0)) for z, var in zip(z_list, var_list)],
        axis=0,
    )
    return np.minimum(hb, ADC_FULL_SCALE)


def normalization_factor(
    W: np.ndarray,
    z_list: Sequence[np.ndarray],
    var_list: Sequence[np.ndarray],
    sigma_num: float = SIGMA_NUM,
    tolerance: float = TOLERANCE,
) -> float:
    """max(f0, f1, f2) per spec §5 -- the factor `W` must be divided by so every one of the
    hardware's three overflow limits holds (at `tolerance` of the true limit). May be < 1 (the
    weight is then scaled UP to use the available dynamic range).

    Returns `np.inf` if `W` is all-zero or contains a non-finite value: callers must treat that
    as a failed qubit (e.g. a dead trace), never divide by it.
    """
    W = np.asarray(W)
    if not np.all(np.isfinite(W)) or np.abs(W).max() == 0:
        return np.inf
    hb = signal_bound(z_list, var_list, sigma_num)
    abs_w = np.abs(W)
    f0 = abs_w.max() / (MAX_WEIGHT * tolerance)
    f1 = (abs_w * hb).max() / (MAX_WEIGHT_ADC_MUL * tolerance)
    f2 = (abs_w * hb).sum() / (MAX_WEIGHT_ADC_MUL_SUM * tolerance)
    return max(f0, f1, f2)


def chunk4(w: np.ndarray) -> np.ndarray:
    """Average every 4 samples -- the OPX integration-weight resolution (spec §5). `len(w)`
    must be a multiple of 4; chunking can only shrink |w|, so the overflow margins computed at
    1 ns resolution still hold after this, but `check_weight_limits` re-verifies them anyway on
    the final (chunked, fixed-point-rounded) arrays.
    """
    w = np.asarray(w)
    if len(w) % 4 != 0:
        raise ValueError(f"chunk4 requires a length that's a multiple of 4, got {len(w)}.")
    return w.reshape(-1, 4).mean(axis=1)


def expand_run_length(entries: Iterable[tuple[float, int]]) -> np.ndarray:
    """Inverse of the (value, length) run-length packing QuAM's integration_weights_function
    returns: expand back into a flat, 1-sample-per-entry array. Used to compare a pulse's
    *currently active* weights (whatever they are) against a freshly computed one at the same
    (1 ns) resolution."""
    entries = list(entries)
    if not entries:
        return np.array([])
    return np.concatenate([np.full(int(length), value) for value, length in entries])


def round_to_fixed_point(values, accuracy: float = WEIGHT_FIXED_POINT_ACCURACY) -> np.ndarray:
    """Round to the OPX's integration-weight fixed-point grid (2**-15), matching what
    `DrachmaReadoutPulse.integration_weights_function` applies before the weights ever reach
    the config -- so overflow margins are checked on the values the hardware will actually see,
    not the pre-rounding ones."""
    values = np.asarray(values)
    return np.round(values / accuracy) * accuracy


def check_weight_limits(
    w_chunked: np.ndarray,
    hb_chunked: np.ndarray,
    tolerance: float = TOLERANCE,
) -> dict:
    """Re-check spec §5's three limits on the FINAL arrays: after chunking to 4 ns *and* after
    rounding to the OPX's fixed-point weight grid. Returns each limit's margin (value / limit;
    <= 1.0 means within bounds) plus which one is closest to binding.
    """
    w_chunked = round_to_fixed_point(w_chunked)
    abs_w = np.abs(w_chunked)
    margins = {
        "weight": float(abs_w.max() / (MAX_WEIGHT * tolerance)),
        "adc_mul": float((abs_w * hb_chunked).max() / (MAX_WEIGHT_ADC_MUL * tolerance)),
        "adc_sum": float((abs_w * hb_chunked).sum() / (MAX_WEIGHT_ADC_MUL_SUM * tolerance)),
    }
    binding_limit = max(margins, key=lambda k: margins[k])
    return {**margins, "binding_limit": binding_limit}


def snr_for_weight(W: np.ndarray, D: np.ndarray, var: np.ndarray) -> float:
    """Predicted SNR of the linear estimator I = Re[sum_t conj(W(t)) * D(t)] against
    independent per-sample noise of variance `var(t)`:

        SNR = |sum_t conj(W) * D| / sqrt(sum_t |W|^2 * var)

    `D` is the (complex, time-varying) signal being detected -- here always
    `env_e - env_g` -- and `var` the pooled per-sample variance. This is invariant to any
    global phase or positive real scale applied to `W` (multiplying W by c*e^{i*phi} scales
    both the numerator and denominator by |c| and leaves the ratio unchanged), which is what
    lets three different weight *shapes* be compared fairly on one footing:

        constant weight (today):            snr_for_weight(np.ones_like(D), D, var)
        real envelope, best global angle:    snr_for_weight(np.abs(D), D, var)
        full complex matched filter:         snr_for_weight(D, D, var)   (or the actual,
                                              possibly variance-weighted, W the node computed)

    NaN-safe: returns 0.0 if the denominator vanishes (e.g. all-zero variance).
    """
    W = np.asarray(W)
    D = np.asarray(D)
    var = np.asarray(var)
    signal = np.abs(np.sum(np.conj(W) * D))
    noise = np.sqrt(np.sum(np.abs(W) ** 2 * var))
    return float(signal / noise) if noise > 0 else 0.0
