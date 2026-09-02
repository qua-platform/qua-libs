"""Qubit-Z freq↔flux helpers for flux-distortion calibrations (17a/c and similar).

Two jobs share the same loaded curves:

1. **Pre-run** — pick a Z-pulse amplitude that places the qubit at a target
   detuning below idle (``flux_amp_from_curve`` / ``resolve_flux_amplitudes``).
2. **Post-run** — invert measured absolute frequency vs time into a signed flux
   step response (``frequency_to_flux_deviation``).

Curves come from prior calibrations:

* **03b** qubit spectroscopy vs Z-flux → ``ds_fit.peak_freq`` (relative to RF)
* **09a** Ramsey vs Z-flux → ``f_qubit_vs_flux`` or unfolded Ramsey frequency

Branch convention: ``"right"`` means flux ≥ idle (sweetspot) flux on the
parabola; ``"left"`` means flux ≤ idle. All returned flux offsets are relative
to idle (ΔΦ), not absolute DAC volts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

Branch = Literal["left", "right"]
_OPPOSITE: dict[str, Branch] = {"left": "right", "right": "left"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_node_data_dict(run_id: int) -> dict:
    """Load the saved Qualibrate node payload (dict) for ``run_id`` from storage."""
    from qualibrate.core.utils.node.content import read_node_data
    from qualibrate.core.utils.node.path_solver import get_node_dir_path
    from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

    base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
    node_dir = get_node_dir_path(run_id, base_path)
    return read_node_data(node_dir, run_id, base_path)


def load_spectroscopy_curve(
    run_id: int, qubit_name: str, qubit_rf_freq: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load freq-vs-Z-flux from a 03b spectroscopy-vs-flux run.

    Uses ``ds_fit.peak_freq`` (fit peak relative to the RF LO / drive) and adds
    ``qubit_rf_freq`` so the returned frequency axis is absolute Hz.

    Parameters
    ----------
    run_id :
        Qualibrate snapshot / run ID of ``03b_qubit_spectroscopy_vs_flux``.
    qubit_name :
        Qubit key in ``ds_fit`` (e.g. ``\"q1\"``).
    qubit_rf_freq :
        Idle / RF frequency in Hz used to convert relative peak → absolute freq.
        Typically ``qubit.xy.RF_frequency``.

    Returns
    -------
    (flux_V, freq_Hz) or None
        Flux-sorted 1-D arrays. ``None`` if the run is missing, has no
        ``peak_freq``, or has fewer than two finite points.
    """
    try:
        data = _read_node_data_dict(run_id)
        ds_fit = data.get("ds_fit")
        if ds_fit is None or "peak_freq" not in ds_fit:
            print(
                f"  WARNING: run #{run_id} has no ds_fit.peak_freq for {qubit_name}; " "cannot load spectroscopy curve"
            )
            return None
        flux = np.asarray(ds_fit.flux_bias.values, dtype=float)
        peak = np.asarray(ds_fit.peak_freq.sel(qubit=qubit_name).values, dtype=float)
        freq = qubit_rf_freq + peak
        mask = np.isfinite(flux) & np.isfinite(freq)
        if mask.sum() < 2:
            print(f"  WARNING: Too few finite peak_freq points for {qubit_name} in run #{run_id}")
            return None
        flux_m, freq_m = flux[mask], freq[mask]
        order = np.argsort(flux_m)
        flux_m, freq_m = flux_m[order], freq_m[order]
        print(
            f"  Loaded spectroscopy curve for {qubit_name} from run #{run_id} "
            f"(ds_fit.peak_freq): {len(flux_m)} pts, "
            f"flux=[{flux_m[0]:.4f}, {flux_m[-1]:.4f}] V"
        )
        return flux_m, freq_m
    except Exception as e:
        print(f"  WARNING: Failed to load spectroscopy curve for {qubit_name} from run #{run_id}: {e}")
        return None


def load_ramsey_curve(
    qubit,
    run_id: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load freq-vs-Z-flux from a 09a Ramsey-vs-flux run for ``qubit``.

    Run ID resolution (first hit wins):

    1. Explicit ``run_id`` argument (node param override).
    2. ``qubit.extras['ramsey_vs_flux_calibration_load_id']`` (written by 09a
       when ``save_load_id=True``).

    Preference order for the frequency axis on ``ds_fit``:

    1. ``f_qubit_vs_flux`` (GHz) → absolute Hz.
    2. Else ``unfolded_frequency`` (GHz), converted to absolute qubit frequency
       via ``RF + artificial_detuning − ramsey_freq``.

    Parameters
    ----------
    qubit :
        QUAM qubit (needs ``.name``, ``.xy.RF_frequency``, optionally ``.extras``).
    run_id :
        Optional Qualibrate run ID override. If ``None``, use extras.

    Returns
    -------
    (flux_V, freq_Hz) or None
        Finite points only. ``None`` if no run ID, all-NaN, or load error.
    """
    if run_id is not None:
        rid: Optional[int] = int(run_id)
    elif hasattr(qubit, "extras"):
        extras_id = qubit.extras.get("ramsey_vs_flux_calibration_load_id")
        rid = int(extras_id) if extras_id is not None else None
    else:
        rid = None
    if rid is None:
        return None

    qubit_name = qubit.name
    qubit_rf_freq = float(qubit.xy.RF_frequency)
    print(f"  Loading Ramsey curve for {qubit_name} from run #{rid}")
    try:
        data = _read_node_data_dict(rid)
        ds_fit = data["ds_fit"]
        flux_bias = ds_fit.flux_bias.values

        if "f_qubit_vs_flux" in ds_fit:
            freq_hz = ds_fit["f_qubit_vs_flux"].sel(qubit=qubit_name).values * 1e9
        else:
            ramsey_freq_hz = ds_fit["unfolded_frequency"].sel(qubit=qubit_name).values * 1e9
            artif_det_hz = 0.0
            if "artificial_detuning" in ds_fit:
                artif_det_hz = float(ds_fit["artificial_detuning"].sel(qubit=qubit_name).values) * 1e6
            freq_hz = qubit_rf_freq + artif_det_hz - ramsey_freq_hz

        mask = ~np.isnan(freq_hz)
        if not np.any(mask):
            print(f"  WARNING: All NaN in Ramsey curve for {qubit_name} from run #{rid}")
            return None
        print(
            f"  Loaded Ramsey curve for {qubit_name}: {mask.sum()} pts, "
            f"flux=[{flux_bias[mask][0]:.4f}, {flux_bias[mask][-1]:.4f}] V, "
            f"freq=[{freq_hz[mask].min()/1e9:.4f}, {freq_hz[mask].max()/1e9:.4f}] GHz"
        )
        return flux_bias[mask], freq_hz[mask]
    except Exception as e:
        print(f"  WARNING: Failed to load Ramsey curve for {qubit_name} from run #{rid}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mapping: detuning → amp (pre-run) and freq → ΔΦ (post-run)
# ---------------------------------------------------------------------------


def flux_amp_from_curve(
    detuning_hz: float,
    idle_freq_hz: float,
    curve: Tuple[np.ndarray, np.ndarray],
    branch: Optional[Branch] = None,
) -> Optional[float]:
    """Invert a freq-vs-flux curve to a Z-pulse amplitude for a target detuning.

    Target absolute frequency is always **below** idle::

        f_target = idle_freq_hz − |detuning_hz|

    Idle flux is the curve sample whose frequency is closest to ``idle_freq_hz``.
    On the selected branch (or full curve), find zero-crossings of
    ``curve_freq − f_target`` and linearly interpolate flux between the two
    flanking points. No spline — discrete crossing + lerp.

    Parameters
    ----------
    detuning_hz :
        Desired |Δf| below idle (Hz). Sign is ignored.
    idle_freq_hz :
        Idle / sweetspot frequency (Hz), usually ``qubit.xy.RF_frequency``.
    curve :
        ``(flux_V, freq_Hz)`` from ``load_spectroscopy_curve`` / ``load_ramsey_curve``.
    branch :
        ``\"right\"`` / ``\"left\"`` — restrict to flux ≥ / ≤ idle flux and return
        the **signed** ΔΦ of the first crossing.
        ``None`` — search the whole curve, pick the crossing with smallest
        |ΔΦ|, return **|ΔΦ|** (used by short-distortion / 17c).

    Returns
    -------
    float or None
        Flux offset from idle in volts, or ``None`` if the branch has < 2 points
        or ``f_target`` never crosses the curve.
    """
    curve_flux, curve_freq = curve
    target_freq = idle_freq_hz - abs(detuning_hz)
    idle_flux = float(curve_flux[int(np.argmin(np.abs(curve_freq - idle_freq_hz)))])

    if branch == "right":
        mask = curve_flux >= idle_flux
        b_flux, b_freq = curve_flux[mask], curve_freq[mask]
    elif branch == "left":
        mask = curve_flux <= idle_flux
        b_flux, b_freq = curve_flux[mask], curve_freq[mask]
    else:
        b_flux, b_freq = curve_flux, curve_freq

    if len(b_flux) < 2:
        return None

    diff = b_freq - target_freq
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings) == 0:
        return None

    def _lerp_delta(i: int) -> float:
        f1, f2 = b_freq[i], b_freq[i + 1]
        x1, x2 = b_flux[i], b_flux[i + 1]
        frac = (target_freq - f1) / (f2 - f1) if abs(f2 - f1) > 0 else 0.0
        return float(x1 + frac * (x2 - x1) - idle_flux)

    if branch is not None:
        return _lerp_delta(int(crossings[0]))

    best = min((_lerp_delta(int(i)) for i in crossings), key=abs)
    return abs(best)


def frequency_to_flux_deviation(
    measured_abs_freq: np.ndarray,
    curve_flux: np.ndarray,
    curve_freq: np.ndarray,
    idle_freq: float,
    use_upper_branch: bool = True,
) -> np.ndarray:
    """Map measured absolute qubit frequency → signed flux deviation from idle.

    Used in analysis after center frequencies vs time are extracted: invert the
    same freq↔flux curve so the cryoscope / π-vs-flux trace becomes a flux step
    response ΔΦ(t).

    Method: mask to the chosen branch (flux ≥ idle if ``use_upper_branch``, else
    ≤), sort by frequency, fit a ``scipy.interpolate.CubicSpline`` of
    flux(freq), evaluate at ``measured_abs_freq``, subtract idle flux.

    Points that land far outside the branch (ΔΦ < −0.01 V or > 2× the branch
    flux span) are set to NaN.

    Parameters
    ----------
    measured_abs_freq :
        Absolute qubit frequency (Hz), any shape; returned array matches shape.
    curve_flux, curve_freq :
        Dispersion curve arrays (same convention as the loaders).
    idle_freq :
        Idle frequency (Hz) used to locate idle flux on the curve.
    use_upper_branch :
        ``True`` → right branch (flux ≥ idle); ``False`` → left.

    Returns
    -------
    np.ndarray
        Signed ΔΦ in volts, same shape as ``measured_abs_freq``. All-NaN if the
        branch has fewer than 4 points.
    """
    from scipy.interpolate import CubicSpline

    idle_idx = int(np.argmin(np.abs(curve_freq - idle_freq)))
    idle_flux = float(curve_flux[idle_idx])

    branch_mask = curve_flux >= idle_flux if use_upper_branch else curve_flux <= idle_flux
    b_flux = curve_flux[branch_mask]
    b_freq = curve_freq[branch_mask]

    if len(b_flux) < 4:
        return np.full_like(measured_abs_freq, np.nan, dtype=float)

    sort_idx = np.argsort(b_freq)
    cs_inv = CubicSpline(b_freq[sort_idx], b_flux[sort_idx], extrapolate=True)
    measured_flat = np.asarray(measured_abs_freq, dtype=float).ravel()
    result = cs_inv(measured_flat) - idle_flux

    flux_range = float(b_flux.max() - idle_flux)
    bad = (result < -0.01) | (result > 2.0 * flux_range)
    result[bad] = np.nan
    return result.reshape(np.shape(measured_abs_freq))


# ---------------------------------------------------------------------------
# Pre-run: spectroscopy → Ramsey → quad_term
# ---------------------------------------------------------------------------


@dataclass
class ResolvedFluxAmps:
    """Result of ``resolve_flux_amplitudes`` for a qubit batch.

    Attributes
    ----------
    amplitudes :
        Signed Z-pulse amplitudes (V) aligned with the input qubit list.
    sources :
        Human-readable origin per qubit, e.g.
        ``\"spectroscopy #42 (right)\"`` or ``\"quad_term=1.2e9\"``.
    effective_branch :
        Branch actually used. May differ from the requested branch if the
        preferred side had no crossing and the opposite side was used.
    """

    amplitudes: List[float]
    sources: List[str]
    effective_branch: Branch

    @property
    def flux_amp_for_detuning_sentinel(self) -> float:
        """±999 sentinel encoding ``effective_branch`` for analysis.

        Analysis historically keyed branch off the sign of a namespace amp
        (positive → right / upper, negative → left). Real amps are overwritten
        elsewhere; this sentinel preserves branch without threading a string
        through the fit path. Prefer passing ``flux_branch`` explicitly when
        possible.
        """
        return 999.0 if self.effective_branch == "right" else -999.0


def resolve_flux_amplitudes(
    qubits,
    *,
    detuning_hz: float,
    flux_branch: Branch,
    use_spectroscopy_data: bool = False,
    spectroscopy_run_id: Optional[int] = None,
    use_ramsey_data: bool = False,
    ramsey_run_id: Optional[int] = None,
) -> ResolvedFluxAmps:
    """Derive per-qubit Z-pulse amplitudes for a target detuning.

    Intended for node setup (e.g. 17a ``create_qua_program``) before the QUA
    program is built. For each qubit, first successful path wins:

    1. **Spectroscopy** (``use_spectroscopy_data`` + ``spectroscopy_run_id``):
       load 03b curve → ``flux_amp_from_curve`` on ``flux_branch``, then
       opposite branch if needed.
    2. **Ramsey** (``use_ramsey_data`` + run id / extras): same on the 09a curve.
    3. **Quadratic fallback**:
       ``sign(flux_branch) * sqrt(|detuning| / |freq_vs_flux_01_quad_term|)``.

    Raises ``ValueError`` if every path fails. Warns if ``|amp| > 0.5`` V.

    Parameters
    ----------
    qubits :
        Iterable of QUAM qubit objects (need ``.name``, ``.xy.RF_frequency``,
        and optionally ``.freq_vs_flux_01_quad_term`` / ``.extras``).
    detuning_hz :
        Target |Δf| below idle (Hz).
    flux_branch :
        Preferred parabola side (``\"left\"`` / ``\"right\"``).
    use_spectroscopy_data, spectroscopy_run_id :
        Enable / identify the 03b source run.
    use_ramsey_data, ramsey_run_id :
        Enable Ramsey path; ``ramsey_run_id`` overrides per-qubit extras
        (see ``load_ramsey_curve``).

    Returns
    -------
    ResolvedFluxAmps
        Amplitudes, source labels, and the effective branch (last qubit's
        branch is stored on the dataclass — fine for single-qubit / shared
        branch runs).
    """
    amplitudes: List[float] = []
    sources: List[str] = []
    effective_branch: Branch = flux_branch

    for q in qubits:
        amp: Optional[float] = None
        label: Optional[str] = None
        used_branch: Branch = flux_branch
        idle = q.xy.RF_frequency

        if use_spectroscopy_data and spectroscopy_run_id is not None:
            curve = load_spectroscopy_curve(spectroscopy_run_id, q.name, idle)
            if curve is not None:
                for br in (flux_branch, _OPPOSITE[flux_branch]):
                    amp = flux_amp_from_curve(detuning_hz, idle, curve, br)
                    if amp is not None:
                        used_branch = br
                        if br != flux_branch:
                            warnings.warn(
                                f"{q.name}: target detuning not found on "
                                f"{flux_branch} branch of spectroscopy "
                                f"#{spectroscopy_run_id}, trying {br}"
                            )
                            label = f"spectroscopy #{spectroscopy_run_id} ({br}, fallback)"
                        else:
                            label = f"spectroscopy #{spectroscopy_run_id} ({br})"
                        break

        if amp is None and use_ramsey_data:
            curve = load_ramsey_curve(q, ramsey_run_id)
            if curve is not None:
                rid_label = f"#{ramsey_run_id}" if ramsey_run_id is not None else "extras"
                for br in (flux_branch, _OPPOSITE[flux_branch]):
                    amp = flux_amp_from_curve(detuning_hz, idle, curve, br)
                    if amp is not None:
                        used_branch = br
                        if br != flux_branch:
                            warnings.warn(
                                f"{q.name}: target detuning not found on "
                                f"{flux_branch} branch of Ramsey {rid_label}, trying {br}"
                            )
                            label = f"Ramsey {rid_label} ({br}, fallback)"
                        else:
                            label = f"Ramsey {rid_label} ({br})"
                        break

        if amp is None:
            qt = getattr(q, "freq_vs_flux_01_quad_term", None)
            if qt is not None and qt != 0 and np.isfinite(qt):
                sign = 1.0 if flux_branch == "right" else -1.0
                amp = sign * float(np.sqrt(abs(detuning_hz) / abs(qt)))
                label = f"quad_term={qt:.3e}"
                used_branch = flux_branch

        if amp is None:
            raise ValueError(
                f"Cannot derive flux_amp for {q.name}: no curve available and "
                f"freq_vs_flux_01_quad_term is missing or zero."
            )

        if abs(amp) > 0.5:
            warnings.warn(
                f"{q.name}: derived flux_amp={amp:.4f} V exceeds 0.5 V. " f"Verify detuning_in_mhz is correct."
            )

        amplitudes.append(float(amp))
        sources.append(label or "unknown")
        effective_branch = used_branch

    return ResolvedFluxAmps(
        amplitudes=amplitudes,
        sources=sources,
        effective_branch=effective_branch,
    )
