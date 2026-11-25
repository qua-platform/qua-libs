"""Drive-line crosstalk measured at the source frequency: observables and complex (bowl) fit.

For a directed pair (source -> victim), the source line plays its own gate at f_source and the victim line plays the
same waveform at f_source scaled by a complex z. The victim then feels a field proportional to (c + z), where c is
the crosstalk coefficient. Both observables are quadratic in that field:

    phase mode       Ramsey phase of the victim (AC Stark shift)   phi = a |c + z|^2 + e
    population mode  excited population of the victim (close pairs) P_e = a |c + z|^2 + e

Expanding, obs = a |z|^2 + b Re(z) + d Im(z) + e', a linear model whose fit gives c = (b + i d) / (2a).
The compensation is to play -c times the source pulse on the victim line, at f_source.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

PHASE_MODE = "phase"
POPULATION_MODE = "population"
# Final frame of the second x90, in turns: the four populations give cos and sin of the Ramsey phase
FRAMES = (0.0, 0.25, 0.5, 0.75)
# A stored coefficient is marked stale when the source frequency has moved by more than this since its measurement
STALE_FREQUENCY_HZ = 5e6

# Acceptance of a fit
MIN_SIGNIFICANCE = 5.0
"""Minimum curvature of the bowl, in standard deviations. The curvature comes from the tone, not from the crosstalk,
so a low one means the measurement itself failed (a weak crosstalk does not fail)."""
MIN_VISIBILITY = 0.2
"""Minimum Ramsey visibility of every grid point (phase mode)."""
MAX_PHASE_RAD = 0.9 * np.pi
"""Maximum spread of the Ramsey phase over the grid (phase mode). The phase is read relative to its circular mean,
which sits near the ring points (most of the grid), so the centre can wrap well before a spread of 2 pi."""
MAX_POPULATION_CHANGE = 0.5
"""Maximum change of the excited population over the grid, relative to the reference point (source and tone off),
in population mode (small-angle regime). Relative, so that a readout offset does not count."""


@dataclass
class CrosstalkFit:
    """Complex drive-line crosstalk coefficient of a single directed (source -> victim) pair."""

    source_qubit: str
    victim_qubit: str
    mode: str
    c_re: float
    c_im: float
    c_err: float
    curvature: float
    curvature_significance: float
    offset: float
    reduced_chi2: float
    frequency_hz: float
    success: bool
    reason: str


# --------------------------------------------------------------------------------------------- observables
def ramsey_phase(pe: np.ndarray, num_shots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ramsey phase, its uncertainty and the visibility of every grid point from the four final frames.

    ``pe`` has shape (point, 4) for the frames (0, 1/4, 1/2, 3/4) turn. The phase is returned relative to the
    circular mean over the points, so that it does not wrap as long as the spread stays within +/- pi. Its sign
    convention is irrelevant for c, which is a ratio of fitted coefficients.
    """
    pe = np.asarray(pe, dtype=float)
    x = pe[:, 2] - pe[:, 0]
    y = pe[:, 3] - pe[:, 1]
    s = x + 1j * y
    reference = np.sum(s)
    reference = reference / abs(reference) if abs(reference) > 0 else 1.0
    phase = np.angle(s * np.conj(reference))
    var = np.clip(pe * (1 - pe), 1.0 / num_shots, None) / num_shots
    sigma = np.sqrt(0.5 * (var[:, 0] + var[:, 1] + var[:, 2] + var[:, 3])) / np.maximum(np.abs(s), 1e-9)
    return phase, sigma, np.abs(s)


def population(pe: np.ndarray, num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Excited population of every grid point and its binomial uncertainty.

    ``pe`` has shape (point, frame). Without Ramsey pulses the final frame does nothing, so the frames are repeats
    of the same measurement and are averaged."""
    pe = np.asarray(pe, dtype=float)
    shots = num_shots * np.sum(np.isfinite(pe), axis=1)
    p = np.nanmean(pe, axis=1)
    sigma = np.sqrt(np.clip(p * (1 - p), 1.0 / shots, None) / shots)
    return p, sigma


# --------------------------------------------------------------------------------------------- bowl fit
def fit_bowl(z: np.ndarray, obs: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    """Weighted linear fit of obs = a |z|^2 + b Re z + d Im z + e and the resulting c = (b + i d) / (2a).

    Returns a, its uncertainty, c, the 1-sigma radius of the uncertainty of c and the reduced chi-squared.
    The covariance is scaled by the reduced chi-squared when the latter exceeds one.
    """
    z = np.asarray(z, dtype=complex)
    design = np.column_stack([np.abs(z) ** 2, z.real, z.imag, np.ones(len(z))])
    w = 1.0 / np.asarray(sigma, dtype=float)
    coef, *_ = np.linalg.lstsq(design * w[:, None], obs * w, rcond=None)
    residual = (obs - design @ coef) * w
    dof = max(len(z) - 4, 1)
    chi2 = float(np.sum(residual**2) / dof)
    cov = np.linalg.pinv((design * w[:, None]).T @ (design * w[:, None])) * max(chi2, 1.0)
    a, b, d, e = coef
    c = (b + 1j * d) / (2 * a)
    # Jacobian of (Re c, Im c) with respect to (a, b, d, e)
    jac = np.array([[-b / (2 * a**2), 1 / (2 * a), 0, 0], [-d / (2 * a**2), 0, 1 / (2 * a), 0]])
    cov_c = jac @ cov @ jac.T
    return {
        "a": float(a),
        "a_err": float(np.sqrt(cov[0, 0])),
        "e": float(e),
        "c": complex(c),
        "c_err": float(np.sqrt(np.trace(cov_c) / 2)),
        "reduced_chi2": chi2,
    }


def model(z: np.ndarray, a: float, c: complex, e0: float) -> np.ndarray:
    """Bowl a |c + z|^2 + e0 (e0 is the value at the null z = -c)."""
    return a * np.abs(c + np.asarray(z)) ** 2 + e0


def analyse_pair(ds_pair: xr.Dataset) -> Tuple[CrosstalkFit, np.ndarray, np.ndarray]:
    """Fit one pair. ``ds_pair`` holds pe(point, frame), z_re/z_im(point), source_on(point) and the pair attrs.

    Returns the fit and the observable with its uncertainty on every grid point (NaN where the source is off)."""
    mode = str(ds_pair.mode.values)
    num_shots = int(ds_pair.attrs["num_shots"])
    on = ds_pair.source_on.values.astype(bool)
    z = (ds_pair.z_re.values + 1j * ds_pair.z_im.values)[on]
    pe = ds_pair.pe.values[on]
    reasons = []
    if mode == PHASE_MODE:
        obs, sigma, visibility = ramsey_phase(pe, num_shots)
        if np.min(visibility) < MIN_VISIBILITY:
            reasons.append(f"Ramsey visibility {np.min(visibility):.2f} < {MIN_VISIBILITY}")
    else:
        obs, sigma = population(pe, num_shots)
        # Relative to the reference point (source and tone off), so that a readout offset does not count
        off = ~on
        reference = population(ds_pair.pe.values[off], num_shots)[0].mean() if off.any() else np.min(obs)
        change = np.max(np.abs(obs - reference))
        if change > MAX_POPULATION_CHANGE:
            reasons.append(
                f"P_e changes by {change:.2f} from the reference > {MAX_POPULATION_CHANGE} (not small-angle)"
            )
    fit = fit_bowl(z, obs, sigma)
    significance = abs(fit["a"]) / fit["a_err"] if fit["a_err"] > 0 else np.inf
    if significance < MIN_SIGNIFICANCE:
        reasons.append(f"curvature {significance:.1f} sigma < {MIN_SIGNIFICANCE}")
    center = complex(ds_pair.center_re.values + 1j * ds_pair.center_im.values)
    # Radius of the pair's grid (adapted per pair), or the node's r_max for datasets without it
    r_max = float(ds_pair["r_max"]) if "r_max" in ds_pair else float(ds_pair.attrs["r_max"])
    if abs(-fit["c"] - center) > 1.5 * r_max:
        reasons.append(
            f"null at {abs(-fit['c'] - center):.3f} from the grid centre, beyond 1.5 r_max (r_max={r_max}): "
            "re-run with center='previous' or a larger r_max"
        )
    if mode == PHASE_MODE and np.max(np.abs(obs - np.min(obs))) > MAX_PHASE_RAD:
        reasons.append(f"Ramsey phase spread above {MAX_PHASE_RAD:.2f} rad (risk of phase wrap)")
    result = CrosstalkFit(
        source_qubit=str(ds_pair.source_qubit.values),
        victim_qubit=str(ds_pair.victim_qubit.values),
        mode=mode,
        c_re=float(fit["c"].real),
        c_im=float(fit["c"].imag),
        c_err=fit["c_err"],
        curvature=fit["a"],
        curvature_significance=float(significance),
        offset=fit["e"] - fit["a"] * abs(fit["c"]) ** 2,
        reduced_chi2=fit["reduced_chi2"],
        frequency_hz=float(ds_pair.source_frequency.values),
        success=not reasons,
        reason="; ".join(reasons),
    )
    obs_all = np.full(on.shape, np.nan)
    sigma_all = np.full(on.shape, np.nan)
    obs_all[on], sigma_all[on] = obs, sigma
    return result, obs_all, sigma_all


OLD_NAMES = {
    "pair": "qubit_pair",
    "drive_qubit": "source_qubit",
    "target_qubit": "victim_qubit",
    "drive_frequency": "source_frequency",
    "target_lo": "victim_lo",
}
"""Names used by datasets written before the source/victim rename, and their current names."""


def upgrade_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename the variables of a dataset written before the source/victim rename (no-op for current datasets)."""
    return ds.rename({old: new for old, new in OLD_NAMES.items() if old in ds.variables or old in ds.dims})


def fit_raw_data(ds: xr.Dataset, node) -> Tuple[xr.Dataset, Dict[str, CrosstalkFit]]:
    """Fit every measured pair of the raw dataset (dimension ``qubit_pair``).

    ds_fit holds, per pair, the fitted coefficient and bowl, and the observable (Stark phase or P_e) with its
    uncertainty on every grid point."""
    ds = upgrade_names(ds)
    fits, obs, sigma = {}, [], []
    for pair in ds.qubit_pair.values:
        fit, obs_pair, sigma_pair = analyse_pair(ds.sel(qubit_pair=pair))
        fits[str(pair)] = fit
        obs.append(obs_pair)
        sigma.append(sigma_pair)
    pairs = list(fits)
    ds_fit = xr.Dataset(
        {
            "obs": (("qubit_pair", "point"), np.array(obs)),
            "obs_err": (("qubit_pair", "point"), np.array(sigma)),
            "z_re": ds.z_re,
            "z_im": ds.z_im,
            "center_re": ds.center_re,
            "center_im": ds.center_im,
        },
        coords={"qubit_pair": pairs, "point": ds.point.values},
    )
    for key in ("c_re", "c_im", "c_err", "curvature", "curvature_significance", "offset", "reduced_chi2", "success"):
        ds_fit[key] = ("qubit_pair", [getattr(fits[p], key) for p in pairs])
    ds_fit["c_abs"] = np.hypot(ds_fit.c_re, ds_fit.c_im)
    ds_fit["c_phase"] = np.arctan2(ds_fit.c_im, ds_fit.c_re)
    ds_fit = ds_fit.assign_coords(source_qubit=ds.source_qubit, victim_qubit=ds.victim_qubit, mode=ds["mode"])
    return ds_fit, fits


def _is_stale(entry: dict, source, victim) -> bool:
    """A stored coefficient is stale when the victim LO changed (the phase of c refers to it) or the source frequency
    moved by more than STALE_FREQUENCY_HZ since it was measured."""
    lo = entry.get("victim_lo_hz", entry.get("target_lo_hz"))  # target_lo_hz: entries written before the rename
    if lo is not None and abs(float(lo) - float(victim.xy.LO_frequency)) > 1.0:
        return True
    frequency = entry.get("frequency_hz")
    return frequency is not None and abs(float(frequency) - float(source.xy.RF_frequency)) > STALE_FREQUENCY_HZ


def matrix_names(machine_qubits, selected: List[str], key: str) -> List[str]:
    """Qubits of the state matrix: the selected ones and every qubit with a stored coefficient (as source or victim),
    in the order of the machine."""
    names = set(selected)
    for name, qubit in machine_qubits.items():
        entries = (qubit.extras or {}).get(key, {})
        if entries:
            names.add(name)
            names.update(entries.keys())
    return [name for name in machine_qubits.keys() if name in names]


def state_matrix(machine_qubits, names: List[str], key: str, fits: Optional[Dict[str, CrosstalkFit]] = None) -> dict:
    """The crosstalk matrix c[victim, source] as the state holds it once this run's results are written.

    Returns the arrays (victim, source): c (complex, NaN where nothing is stored), c_err, stale (stored entry whose
    victim LO or source frequency changed since), updated (written by this run) and failed (measured by this run
    without a passing fit, so the stored value, if any, is kept)."""
    n = len(names)
    matrix = {
        "c": np.full((n, n), np.nan, dtype=complex),
        "c_err": np.full((n, n), np.nan),
        "stale": np.zeros((n, n), dtype=bool),
        "updated": np.zeros((n, n), dtype=bool),
        "failed": np.zeros((n, n), dtype=bool),
    }
    for i, source_name in enumerate(names):
        source = machine_qubits[source_name]
        for victim_name, entry in (source.extras or {}).get(key, {}).items():
            if victim_name not in names:
                continue
            j = names.index(victim_name)
            matrix["c"][j, i] = complex(entry["c_re"], entry["c_im"])
            matrix["c_err"][j, i] = entry.get("c_err", np.nan)
            matrix["stale"][j, i] = _is_stale(entry, source, machine_qubits[victim_name])
    for fit in (fits or {}).values():
        if fit.source_qubit not in names or fit.victim_qubit not in names:
            continue
        j, i = names.index(fit.victim_qubit), names.index(fit.source_qubit)
        if fit.success:
            matrix["c"][j, i] = complex(fit.c_re, fit.c_im)
            matrix["c_err"][j, i] = fit.c_err
            matrix["stale"][j, i] = False
            matrix["updated"][j, i] = True
        else:
            matrix["failed"][j, i] = True
    return matrix


def log_fitted_results(fit_results: Dict[str, dict], log_callable=None):
    """Log c, its uncertainty and the verdict of every measured pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for pair, fit in fit_results.items():
        c = complex(fit["c_re"], fit["c_im"])
        verdict = "SUCCESS!" if fit["success"] else f"FAIL! ({fit['reason']})"
        log_callable(
            f"Drive-line crosstalk {pair} [{fit['mode']}]: |c| = {100 * abs(c):.2f} +/- {100 * fit['c_err']:.2f} % of "
            f"amplitude, phase {np.rad2deg(np.angle(c)):.1f} deg, curvature {fit['curvature_significance']:.1f} sigma, "
            f"chi2_r = {fit['reduced_chi2']:.2f} --> {verdict}"
        )


def stored_coefficient(source_extras: dict, victim: str, key: str) -> Optional[complex]:
    """The complex coefficient stored on the source qubit for ``victim``, if any."""
    entry = (source_extras or {}).get(key, {}).get(victim)
    if not entry:
        return None
    return complex(entry["c_re"], entry["c_im"])
