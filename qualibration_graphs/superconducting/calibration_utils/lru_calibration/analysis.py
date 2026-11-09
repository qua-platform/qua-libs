"""Analysis of the f-LRU calibration: resonance line in (amplitude, frequency) and P(f) along it.

For each qubit and LRU duration the resonance appears as a dip in P(f) whose frequency moves with
the drive amplitude A through the AC Stark shift,

    detuning(A) = d0 + c2 * A**2.

The resonance is located per amplitude from the minimum of P(f) along frequency, the quadratic
line is fitted through these points, and P(f) is read off along the fitted line. The optimum is
the amplitude of lowest P(f) on the line; across durations the one reaching the lowest P(f) wins.

The population that rises at the resonance depends on the transition, which the raw dataset carries
in ``attrs["lru_transition"]``: P(g) for |f0> -> |g1> (node 26a), P(e) for |f0> -> |e1> (node 26b).
The fit uses P(f) only.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from quam_config.lru_components import LEGACY_LRU_TRANSITION, LRU_TRANSITIONS, LruTransition

#: Smallest number of amplitudes with a visible dip needed to fit the resonance line.
MIN_RIDGE_POINTS = 3
#: P(f) on the line must drop below this fraction of the prepared P(f) for a fit to succeed.
MAX_RESIDUAL_FRACTION = 0.5


@dataclass
class FitParameters:
    """Calibrated LRU pump for one qubit (NaN fields when no duration succeeded)."""

    success: bool
    duration_ns: float
    amplitude: float
    amp_prefactor: float
    frequency: float
    frequency_zero_amplitude: float
    stark_coefficient: float
    pf_min: float
    pf_prepared: float
    failure_reason: str = ""
    transition: str = ""


def lru_transition(ds: xr.Dataset) -> LruTransition:
    """Transition stored with the dataset; ``LEGACY_LRU_TRANSITION`` when the tag is missing or unknown."""
    tag = ds.attrs.get("lru_transition")
    return tag if tag in LRU_TRANSITIONS else LEGACY_LRU_TRANSITION


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add P(g) to the populations measured by the g/e/f readout."""
    ds = ds.assign(pg=1 - ds.pe - ds.pf)
    ds.pg.attrs = {"long_name": "P(g)"}
    ds.pe.attrs = {"long_name": "P(e)"}
    ds.pf.attrs = {"long_name": "P(f)"}
    return ds


def assign_lru_coords(ds: xr.Dataset, qubits, operation: str) -> xr.Dataset:
    """Add the absolute LRU amplitude and drive frequency, and the |f0> -> |g1> tag, stored with the raw data.

    They are taken from the LRU elements at acquisition time, since an element added for the run only
    is gone from the saved QUAM state when the data is reloaded.
    """
    base_amp = np.array([q.lru.operations[operation].amplitude for q in qubits])
    rf = np.array([q.lru.RF_frequency for q in qubits])
    ds = ds.assign_coords(
        amplitude=(
            ("qubit", "amp_prefactor"),
            base_amp[:, None] * ds.amp_prefactor.values[None, :],
            {"long_name": "LRU amplitude", "units": "fraction of full scale"},
        ),
        frequency=(
            ("qubit", "detuning"),
            rf[:, None] + ds.detuning.values[None, :],
            {"long_name": "LRU drive frequency", "units": "Hz"},
        ),
    )
    ds.attrs["lru_transition"] = "f0g1"
    return ds


def assign_flux_lru_coords(ds: xr.Dataset, qubits, amplitude_v: float, centres: np.ndarray) -> xr.Dataset:
    """Add the flux-tone amplitude (V), the |f0> -> |e1> drive frequency (twice the flux tone) and the f0e1 tag."""
    ds = ds.assign_coords(
        amplitude=(
            ("qubit", "amp_prefactor"),
            np.tile(amplitude_v * ds.amp_prefactor.values, (len(qubits), 1)),
            {"long_name": "LRU flux-tone amplitude", "units": "V"},
        ),
        frequency=(
            ("qubit", "detuning"),
            np.asarray(centres)[:, None] + ds.detuning.values[None, :],
            {"long_name": "|f0> -> |e1> drive frequency (2 x flux tone)", "units": "Hz"},
        ),
    )
    ds.attrs["lru_transition"] = "f0e1"
    return ds


def _ridge(detuning: np.ndarray, pf: np.ndarray, min_contrast: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resonance detuning per amplitude from the P(f) minimum, and which amplitudes show a dip.

    ``pf`` has shape (amplitude, detuning). Minima on the window edge are rejected, since the
    resonance may lie outside the sweep.
    """
    kernel = np.ones(3) / 3
    smooth = np.array([np.convolve(np.pad(row, 1, mode="edge"), kernel, mode="valid") for row in pf])
    baseline = np.nanpercentile(pf, 90, axis=1)
    reference = np.nanmax(baseline)
    idx = np.nanargmin(smooth, axis=1)
    depth = baseline - smooth[np.arange(len(idx)), idx]
    valid = (depth > min_contrast * reference) & (idx > 0) & (idx < len(detuning) - 1)

    ridge = np.full(len(idx), np.nan)
    step = detuning[1] - detuning[0]
    for i in np.flatnonzero(valid):
        k = idx[i]
        y0, y1, y2 = smooth[i, k - 1 : k + 2]
        curvature = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / curvature if curvature > 0 else 0.0
        ridge[i] = detuning[k] + np.clip(shift, -1, 1) * step
    return ridge, valid


def _line_index(detuning: np.ndarray, line: np.ndarray, pf: np.ndarray) -> np.ndarray:
    """Per amplitude, the detuning sample read as "on the line" (-1 when the line leaves the window).

    It is the lower-P(f) of the two samples bracketing the line: interpolating between them would
    average away a resonance narrower than the frequency step.
    """
    upper = np.clip(np.searchsorted(detuning, line), 1, len(detuning) - 1)
    lower = upper - 1
    rows = np.arange(len(line))
    index = np.where(pf[rows, lower] <= pf[rows, upper], lower, upper)
    inside = (line >= detuning.min()) & (line <= detuning.max())
    return np.where(inside, index, -1)


def fit_lru_map(
    detuning: np.ndarray,
    amplitude: np.ndarray,
    pf: np.ndarray,
    num_shots: int,
    min_contrast: float = 0.2,
    min_prepared_pf: float = 0.5,
    min_dip_sigma: float = 5.0,
) -> Dict:
    """Fit the resonance line ``d0 + c2 * A**2`` of one P(f) map and read P(f) along it.

    Args:
        detuning: Drive detunings in Hz, shape (n_det,).
        amplitude: Absolute LRU amplitudes, shape (n_amp,).
        pf: P(f), shape (n_amp, n_det).
        num_shots: Shots averaged per point, which sets the shot noise of P(f).
        min_contrast: Minimum dip, as a fraction of the prepared P(f), for an amplitude to be used.
        min_prepared_pf: Minimum P(f) at the lowest amplitude; below it the map is not fitted.
        min_dip_sigma: Minimum drop of P(f) on the line, in shot-noise standard deviations.

    Returns:
        Dict with the per-amplitude ``ridge``, ``valid``, ``line`` (fitted detuning), ``line_index``
        (detuning sample on the line, see ``_line_index``) and ``pf_line``,
        the fit coefficients ``d0`` and ``c2``, the optimum ``opt_index``, ``pf_min``,
        ``pf_prepared``, ``success`` and, on failure, ``failure_reason``. Fit quantities are NaN
        when the map is not fitted or too few amplitudes show a dip.
    """
    n_amp = len(amplitude)
    # Median, not a high percentile: this is the |f> population the LRU starts from, not a baseline for dips.
    pf_prepared = float(np.nanmedian(pf[np.argmin(np.abs(amplitude))]))
    result = dict(
        ridge=np.full(n_amp, np.nan),
        valid=np.zeros(n_amp, dtype=bool),
        line=np.full(n_amp, np.nan),
        line_index=np.full(n_amp, -1),
        pf_line=np.full(n_amp, np.nan),
        d0=np.nan,
        c2=np.nan,
        opt_index=-1,
        pf_min=np.nan,
        pf_prepared=pf_prepared,
        success=False,
        failure_reason="",
    )
    if pf_prepared < min_prepared_pf:
        result["failure_reason"] = (
            f"|f> is not prepared: P(f) = {pf_prepared:.3f} at the lowest LRU amplitude, below min_prepared_pf = "
            f"{min_prepared_pf}. Check x180, EF_x180 and the g/e/f readout before calibrating the LRU. "
            "At long pulse durations, T1 decay of |f> also lowers P(f)."
        )
        return result

    ridge, valid = _ridge(detuning, pf, min_contrast)
    result.update(ridge=ridge, valid=valid)
    use = valid.copy()
    if use.sum() < MIN_RIDGE_POINTS:
        result["failure_reason"] = (
            f"no resonance found: only {int(use.sum())} amplitudes show a P(f) dip deeper than "
            f"min_ridge_contrast = {min_contrast} of the prepared P(f) (need {MIN_RIDGE_POINTS})."
        )
        return result
    c2, d0 = np.polyfit(amplitude[use] ** 2, ridge[use], 1)
    # One pass of outlier rejection: drop points further than 3 MADs from the first fit.
    residuals = ridge - (d0 + c2 * amplitude**2)
    mad = np.nanmedian(np.abs(residuals[use]))
    if mad > 0:
        use &= np.abs(residuals) <= 3 * 1.4826 * mad
        if use.sum() >= MIN_RIDGE_POINTS:
            c2, d0 = np.polyfit(amplitude[use] ** 2, ridge[use], 1)

    line = d0 + c2 * amplitude**2
    line_index = _line_index(detuning, line, pf)
    pf_line = np.array([pf[i, k] if k >= 0 else np.nan for i, k in enumerate(line_index)])
    if not np.any(np.isfinite(pf_line)):
        result["failure_reason"] = "the fitted resonance line leaves the frequency window at every amplitude."
        return result
    opt = int(np.nanargmin(pf_line))
    pf_min = float(pf_line[opt])
    result.update(
        valid=use,
        line=line,
        line_index=line_index,
        pf_line=pf_line,
        d0=float(d0),
        c2=float(c2),
        opt_index=opt,
        pf_min=pf_min,
    )

    # Binomial shot noise of one P(f) point; the minimum over the line is biased low by a few of these.
    sigma = np.sqrt(max(pf_prepared * (1 - pf_prepared), 1 / num_shots) / num_shots)
    if pf_prepared - pf_min < min_dip_sigma * sigma:
        result["failure_reason"] = (
            f"the P(f) dip on the line ({pf_prepared:.3f} -> {pf_min:.3f}) is within {min_dip_sigma:g} times the "
            f"shot noise ({sigma:.3f} at {num_shots} shots): no significant LRU effect."
        )
    elif pf_min >= MAX_RESIDUAL_FRACTION * pf_prepared:
        result["failure_reason"] = (
            f"P(f) on the line only drops from {pf_prepared:.3f} to {pf_min:.3f}, not below "
            f"{MAX_RESIDUAL_FRACTION:.0%} of the prepared P(f)."
        )
    else:
        result["success"] = True
    return result


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit every (qubit, duration) map and pick, per qubit, the duration reaching the lowest P(f)."""
    params = node.parameters
    detuning = ds.detuning.values
    shape = (ds.sizes["qubit"], ds.sizes["duration"], ds.sizes["amp_prefactor"])
    per_amp = {
        key: np.full(shape, np.nan) for key in ("ridge_detuning", "line_detuning", "pf_line", "pe_line", "pg_line")
    }
    ridge_used = np.zeros(shape, dtype=bool)
    per_duration = {key: np.full(shape[:2], np.nan) for key in ("d0", "c2", "pf_min", "pf_prepared", "opt_amplitude")}
    success = np.zeros(shape[:2], dtype=bool)
    opt_index = np.full(shape[:2], -1)
    reasons = [["" for _ in range(shape[1])] for _ in range(shape[0])]

    transition = lru_transition(ds)
    fit_results = {}
    for iq, qubit in enumerate(ds.qubit.values):
        amplitude = ds.amplitude.sel(qubit=qubit).values
        for it, duration in enumerate(ds.duration.values):
            sel = ds[["pf", "pe"]].sel(qubit=qubit, duration=duration).transpose("amp_prefactor", "detuning")
            fit = fit_lru_map(
                detuning,
                amplitude,
                sel.pf.values,
                params.num_shots,
                min_contrast=params.min_ridge_contrast,
                min_prepared_pf=params.min_prepared_pf,
                min_dip_sigma=params.min_dip_sigma,
            )
            per_amp["ridge_detuning"][iq, it] = fit["ridge"]
            per_amp["line_detuning"][iq, it] = fit["line"]
            per_amp["pf_line"][iq, it] = fit["pf_line"]
            pe = sel.pe.values
            per_amp["pe_line"][iq, it] = [pe[i, k] if k >= 0 else np.nan for i, k in enumerate(fit["line_index"])]
            ridge_used[iq, it] = fit["valid"]
            for key in ("d0", "c2", "pf_min", "pf_prepared"):
                per_duration[key][iq, it] = fit[key]
            if fit["opt_index"] >= 0:
                per_duration["opt_amplitude"][iq, it] = amplitude[fit["opt_index"]]
            success[iq, it] = fit["success"]
            opt_index[iq, it] = fit["opt_index"]
            reasons[iq][it] = fit["failure_reason"]

        fit_results[str(qubit)] = _best_duration(ds, iq, per_duration, success, opt_index, reasons[iq], transition)
    # P(g) on the line from the two measured populations; the raw dataset need not hold pg
    per_amp["pg_line"] = 1 - per_amp["pe_line"] - per_amp["pf_line"]

    dims = ("qubit", "duration", "amp_prefactor")
    ds_fit = xr.Dataset(
        {
            **{key: (dims, value) for key, value in per_amp.items()},
            "ridge_used": (dims, ridge_used),
            **{key: (dims[:2], value) for key, value in per_duration.items()},
            "success": (dims[:2], success),
        },
        coords={
            "qubit": ds.qubit,
            "duration": ds.duration,
            "amp_prefactor": ds.amp_prefactor,
            "amplitude": ds.amplitude,
        },
    )
    return ds_fit, fit_results


def _best_duration(
    ds: xr.Dataset,
    iq: int,
    per_duration: Dict,
    success: np.ndarray,
    opt_index: np.ndarray,
    reasons: list,
    transition: str,
):
    """FitParameters of the successful duration with the lowest P(f) on the line for qubit ``iq``.

    When no duration succeeds, the one with the lowest P(f) on the line is reported, with its failure reason.
    """
    pf_min = np.where(success[iq], per_duration["pf_min"][iq], np.inf)
    if not np.isfinite(pf_min).any():
        it = int(np.argmin(np.nan_to_num(per_duration["pf_min"][iq], nan=np.inf)))
        return FitParameters(
            success=False,
            duration_ns=float(ds.duration.values[it]),
            amplitude=float(per_duration["opt_amplitude"][iq, it]),
            amp_prefactor=np.nan,
            frequency=np.nan,
            frequency_zero_amplitude=np.nan,
            stark_coefficient=float(per_duration["c2"][iq, it]),
            pf_min=float(per_duration["pf_min"][iq, it]),
            pf_prepared=float(per_duration["pf_prepared"][iq, it]),
            failure_reason=f"{ds.duration.values[it]:.0f} ns: {reasons[it]}",
            transition=transition,
        )
    it = int(np.argmin(pf_min))
    ia = int(opt_index[iq, it])
    d0, c2 = per_duration["d0"][iq, it], per_duration["c2"][iq, it]
    amplitude = float(ds.amplitude.values[iq, ia])
    rf = float(ds.frequency.values[iq, 0] - ds.detuning.values[0])
    return FitParameters(
        success=True,
        duration_ns=float(ds.duration.values[it]),
        amplitude=amplitude,
        amp_prefactor=float(ds.amp_prefactor.values[ia]),
        frequency=rf + d0 + c2 * amplitude**2,
        frequency_zero_amplitude=rf + d0,
        stark_coefficient=float(c2),
        pf_min=float(per_duration["pf_min"][iq, it]),
        pf_prepared=float(per_duration["pf_prepared"][iq, it]),
        transition=transition,
    )


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the calibrated LRU pump of every qubit."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit, fit in fit_results.items():
        if not fit["success"]:
            log_callable(f"{qubit}: FAIL -- {fit['failure_reason']}")
            continue
        transition = fit.get("transition")
        zero_amp = LRU_TRANSITIONS[transition].label if transition in LRU_TRANSITIONS else "resonance"
        log_callable(
            f"{qubit}: SUCCESS -- tau = {fit['duration_ns']:.0f} ns, amplitude = {fit['amplitude']:.4f} "
            f"(pre-factor {fit['amp_prefactor']:.3f}), f_LRU = {fit['frequency'] / 1e9:.5f} GHz "
            f"({zero_amp} at zero amplitude {fit['frequency_zero_amplitude'] / 1e9:.5f} GHz, "
            f"Stark coefficient {fit['stark_coefficient'] / 1e6:+.1f} MHz/amp^2), "
            f"P(f) {fit['pf_prepared']:.3f} -> {fit['pf_min']:.3g}."
        )


#: A center is refit only if at least this fraction of the shots is nearest to it at the start; an almost
#: empty state (e.g. |e> after a good |f> preparation) would otherwise drift into a neighbouring blob.
MIN_REFIT_FRACTION = 0.05


def _refit_centers(iq: np.ndarray, centers: np.ndarray, iterations: int = 20) -> np.ndarray:
    """k-means on the (n, 2) shots, started from ``centers``; underpopulated states keep their center."""
    centers = centers.copy()
    labels = ((iq[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    movable = [k for k in range(len(centers)) if np.mean(labels == k) >= MIN_REFIT_FRACTION]
    for _ in range(iterations):
        for k in movable:
            if np.any(labels == k):
                centers[k] = iq[labels == k].mean(0)
        labels = ((iq[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    return centers


def classify_iq_shots(ds: xr.Dataset, qubits) -> Tuple[xr.Dataset, Dict[str, Dict]]:
    """Debug mode (``return_iq_blobs``): classify every raw I/Q shot offline and average into P(e) and P(f).

    The g/e/f centers are refit on this run's own shots (k-means started from the stored ``gef_centers``;
    a state holding under ``MIN_REFIT_FRACTION`` of the shots keeps its stored center),
    and each shot gets the state of the nearest refit center. That ``state`` is averaged over shots into the
    ``pe`` and ``pf`` the rest of the analysis uses. ``stored_state`` keeps what the real-time readout would
    give (nearest stored center in L1 distance), for comparison.

    Returns:
        The dataset with ``state``, ``stored_state``, ``pe`` and ``pf``, and per qubit the populations from
        both classifications with the stored and refit centers (raw demodulation units).
    """
    by_name = {q.name: q for q in qubits}
    states, stored_states, info = [], [], {}
    for name in ds.qubit.values:
        stored = np.asarray(by_name[str(name)].resonator.gef_centers, dtype=float)
        shots = ds.sel(qubit=name)
        i_shots, q_shots = shots.I.values, shots.Q.values
        iq = np.stack([i_shots.ravel(), q_shots.ravel()], axis=1)
        refit = _refit_centers(iq, stored)
        state = ((i_shots[..., None] - refit[:, 0]) ** 2 + (q_shots[..., None] - refit[:, 1]) ** 2).argmin(-1)
        stored_state = (np.abs(i_shots[..., None] - stored[:, 0]) + np.abs(q_shots[..., None] - stored[:, 1])).argmin(
            -1
        )
        states.append(state)
        stored_states.append(stored_state)
        info[str(name)] = {
            "populations_refit": [float(np.mean(state == k)) for k in range(3)],
            "populations_stored": [float(np.mean(stored_state == k)) for k in range(3)],
            "centers_stored": stored.tolist(),
            "centers_refit": refit.tolist(),
        }
    ds = ds.assign(state=(ds.I.dims, np.stack(states)), stored_state=(ds.I.dims, np.stack(stored_states)))
    return ds.assign(pe=(ds.state == 1).mean("shot"), pf=(ds.state == 2).mean("shot")), info


def log_iq_results(info: Dict, log_callable=None):
    """Log the populations the stored and the refit centers give to the raw shots of an IQ debug run."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit, r in info.items():
        stored, refit = r["populations_stored"], r["populations_refit"]
        shift = np.linalg.norm(np.subtract(r["centers_refit"], r["centers_stored"]), axis=1)
        log_callable(
            f"{qubit}: IQ debug -- all shots read g/e/f {stored[0]:.3f}/{stored[1]:.3f}/{stored[2]:.3f} with the "
            f"stored centers (real-time rule) and {refit[0]:.3f}/{refit[1]:.3f}/{refit[2]:.3f} with centers refit "
            f"on this run; refit centers moved by {', '.join(f'{d:.1e}' for d in shift)} raw units (g, e, f)."
        )
