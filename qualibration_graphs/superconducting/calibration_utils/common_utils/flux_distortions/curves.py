"""Qubit-Z freq↔flux helpers for flux-distortion calibrations (17a/c and similar).

Two jobs share the same loaded curves:

1. **Pre-run** — pick a Z-pulse amplitude that places the qubit at a target
   detuning below idle (``flux_amp_from_curve`` / ``resolve_flux_amplitudes``).
2. **Post-run** — invert measured absolute frequency vs time into a signed flux
   step response (``frequency_to_flux_deviation``).

Curves come from prior calibrations, and their run IDs are always read from the
QUAM state (``qubit.extras``) — never entered by hand:

* **09a** Ramsey vs Z-flux → ``f_qubit_vs_flux`` or unfolded Ramsey frequency,
  run ID in ``extras['ramsey_vs_flux_calibration_load_id']``
* **03b** qubit spectroscopy vs Z-flux → ``ds_fit.peak_freq`` (relative to RF),
  run ID in ``extras['qubit_spectroscopy_vs_flux_load_id']``

A single node parameter picks the freq→flux source (see
``resolve_freq_flux_curve``): ``"auto"`` (default) tries Ramsey, then
spectroscopy, then the quadratic ``freq_vs_flux_01_quad_term``; the other
values force one specific source.

Branch convention: ``"right"`` means flux ≥ idle (sweetspot) flux on the
parabola; ``"left"`` means flux ≤ idle. All returned flux offsets are relative
to idle (ΔΦ), not absolute DAC volts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

Branch = Literal["left", "right"]
_OPPOSITE: dict[str, Branch] = {"left": "right", "right": "left"}

#: Which freq↔flux relation to use. ``"auto"`` walks ``AUTO_SOURCE_ORDER`` and
#: falls back to ``freq_vs_flux_01_quad_term``; the other values force one source
#: and warn (instead of silently degrading) when it cannot be loaded.
FreqFluxSource = Literal["auto", "ramsey", "spectroscopy", "quad_term"]

#: Priority order used by ``source="auto"``: measured curves first (Ramsey is the
#: most accurate freq↔flux map), quadratic term only as a last resort.
AUTO_SOURCE_ORDER: Tuple[str, ...] = ("ramsey", "spectroscopy")

#: ``qubit.extras`` keys written by the source calibrations.
RAMSEY_EXTRAS_KEY = "ramsey_vs_flux_calibration_load_id"  # 09a, when save_load_id=True
SPECTROSCOPY_EXTRAS_KEY = "qubit_spectroscopy_vs_flux_load_id"  # 03b, when save_load_id=True


def extras_run_id(qubit, key: str) -> Optional[int]:
    """Read a run ID from ``qubit.extras[key]``, or ``None`` if absent/invalid."""
    extras = getattr(qubit, "extras", None)
    if not extras:
        return None
    try:
        value = extras.get(key)
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"  WARNING: {getattr(qubit, 'name', '?')}: extras['{key}']={value!r} is not a run ID")
        return None


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


def load_spectroscopy_curve_for_qubit(qubit) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the 03b freq-vs-Z-flux curve for ``qubit`` using its extras run ID.

    Reads ``qubit.extras['qubit_spectroscopy_vs_flux_load_id']`` (written by 03b
    when ``save_load_id=True``) and delegates to ``load_spectroscopy_curve``.

    Returns
    -------
    (flux_V, freq_Hz) or None
        ``None`` if the qubit has no recorded 03b run or the run cannot be read.
    """
    rid = extras_run_id(qubit, SPECTROSCOPY_EXTRAS_KEY)
    if rid is None:
        return None
    return load_spectroscopy_curve(rid, qubit.name, float(qubit.xy.RF_frequency))


def load_ramsey_curve(qubit) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load freq-vs-Z-flux from a 09a Ramsey-vs-flux run for ``qubit``.

    The run ID comes from ``qubit.extras['ramsey_vs_flux_calibration_load_id']``
    (written by 09a when ``save_load_id=True``).

    Preference order for the frequency axis on ``ds_fit``:

    1. ``f_qubit_vs_flux`` (GHz) → absolute Hz.
    2. Else ``unfolded_frequency`` (GHz), converted to absolute qubit frequency
       via ``RF + artificial_detuning − ramsey_freq``.

    Parameters
    ----------
    qubit :
        QUAM qubit (needs ``.name``, ``.xy.RF_frequency``, optionally ``.extras``).

    Returns
    -------
    (flux_V, freq_Hz) or None
        Finite points only. ``None`` if no run ID, all-NaN, or load error.
    """
    rid = extras_run_id(qubit, RAMSEY_EXTRAS_KEY)
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
# Source selection: one place that decides which freq↔flux relation is used
# ---------------------------------------------------------------------------


@dataclass
class FreqFluxCurve:
    """The freq↔flux relation selected for one qubit.

    Attributes
    ----------
    kind :
        ``"ramsey"`` / ``"spectroscopy"`` — a measured ``curve`` is available.
        ``"quad_term"`` — no curve; use the quadratic ``quad_term``.
        ``"none"`` — nothing usable for this qubit.
    label :
        Human-readable origin for logs and figure titles, e.g.
        ``"Ramsey #42"`` or ``"quad_term=1.200e+09"``.
    curve :
        ``(flux_V, freq_Hz)`` when ``kind`` is a measured source, else ``None``.
    run_id :
        Run ID the curve came from, when applicable.
    quad_term :
        ``freq_vs_flux_01_quad_term`` when ``kind == "quad_term"``.
    """

    kind: Literal["ramsey", "spectroscopy", "quad_term", "none"]
    label: str
    curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
    run_id: Optional[int] = None
    quad_term: Optional[float] = None

    @property
    def is_measured(self) -> bool:
        """True when a measured freq-vs-flux ``curve`` is available."""
        return self.curve is not None


def _load_source_curve(qubit, kind: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the curve for one source ``kind``, or ``None`` if unavailable."""
    if kind == "ramsey":
        return load_ramsey_curve(qubit)
    if kind == "spectroscopy":
        return load_spectroscopy_curve_for_qubit(qubit)
    return None


def _source_run_id(qubit, kind: str) -> Optional[int]:
    """Extras run ID backing source ``kind`` for ``qubit``."""
    if kind == "ramsey":
        return extras_run_id(qubit, RAMSEY_EXTRAS_KEY)
    if kind == "spectroscopy":
        return extras_run_id(qubit, SPECTROSCOPY_EXTRAS_KEY)
    return None


def resolve_freq_flux_curve(qubit, source: FreqFluxSource = "auto") -> FreqFluxCurve:
    """Pick the freq↔flux relation for one qubit.

    This is the **single** decision point for the freq→voltage method, shared by
    the pre-run amplitude resolution and the post-run flux inversion so both
    always agree. Run IDs are read from ``qubit.extras``; the user only chooses
    ``source``.

    Behaviour
    ---------
    * ``"auto"`` — try each source in ``AUTO_SOURCE_ORDER`` (Ramsey, then
      spectroscopy), then fall back to ``freq_vs_flux_01_quad_term``.
    * ``"ramsey"`` / ``"spectroscopy"`` — use only that source. If its extras run
      ID is missing or the run cannot be read, warn loudly and fall back to
      ``quad_term`` rather than failing silently.
    * ``"quad_term"`` — skip curve loading entirely.

    Parameters
    ----------
    qubit :
        QUAM qubit (needs ``.name``, ``.xy.RF_frequency``, optionally ``.extras``
        and ``.freq_vs_flux_01_quad_term``).
    source :
        Requested freq→flux source.

    Returns
    -------
    FreqFluxCurve
        ``kind == "none"`` when neither a curve nor a usable quadratic term
        exists; callers decide whether that is fatal.
    """
    if source == "quad_term":
        candidates: Tuple[str, ...] = ()
    elif source == "auto":
        candidates = AUTO_SOURCE_ORDER
    else:
        candidates = (source,)

    for kind in candidates:
        curve = _load_source_curve(qubit, kind)
        if curve is not None:
            rid = _source_run_id(qubit, kind)
            pretty = "Ramsey" if kind == "ramsey" else "spectroscopy"
            return FreqFluxCurve(kind=kind, label=f"{pretty} #{rid}", curve=curve, run_id=rid)
        if source != "auto":
            extras_key = RAMSEY_EXTRAS_KEY if kind == "ramsey" else SPECTROSCOPY_EXTRAS_KEY
            warnings.warn(
                f"{qubit.name}: freq_to_flux_source='{kind}' was requested but no usable curve "
                f"could be loaded (extras['{extras_key}'] missing or unreadable). Falling back to "
                f"freq_vs_flux_01_quad_term — re-run the source calibration with save_load_id=True."
            )

    qt = getattr(qubit, "freq_vs_flux_01_quad_term", None)
    if qt is not None and qt != 0 and np.isfinite(qt):
        return FreqFluxCurve(kind="quad_term", label=f"quad_term={float(qt):.3e}", quad_term=float(qt))

    return FreqFluxCurve(kind="none", label="unavailable")


def resolve_freq_flux_curves(qubits, source: FreqFluxSource = "auto") -> Dict[str, FreqFluxCurve]:
    """``resolve_freq_flux_curve`` for a qubit batch, keyed by qubit name."""
    return {q.name: resolve_freq_flux_curve(q, source) for q in qubits}


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
        ``\"Ramsey #42 (right)\"`` or ``\"quad_term=1.2e9\"``.
    curves :
        The ``FreqFluxCurve`` chosen per qubit, keyed by qubit name — the same
        selection analysis will make, so it can be reported to the user.
    effective_branch :
        Branch actually used. May differ from the requested branch if the
        preferred side had no crossing and the opposite side was used.
    """

    amplitudes: List[float]
    sources: List[str]
    effective_branch: Branch
    curves: Dict[str, FreqFluxCurve] = field(default_factory=dict)

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
    freq_to_flux_source: FreqFluxSource = "auto",
) -> ResolvedFluxAmps:
    """Derive per-qubit Z-pulse amplitudes for a target detuning.

    Intended for node setup (e.g. 17a ``create_qua_program``) before the QUA
    program is built. The freq↔flux relation is chosen once per qubit by
    ``resolve_freq_flux_curve`` — the same call analysis makes — so the amplitude
    and the later flux inversion always use the same relation.

    Given that relation:

    * **Measured curve** (Ramsey / spectroscopy) → ``flux_amp_from_curve`` on
      ``flux_branch``, falling back to the opposite branch if the target
      detuning has no crossing there.
    * **Quadratic** → ``sign(flux_branch) * sqrt(|detuning| / |quad_term|)``.

    Raises ``ValueError`` if no relation is usable for a qubit. Warns if
    ``|amp| > 0.5`` V.

    Parameters
    ----------
    qubits :
        Iterable of QUAM qubit objects (need ``.name``, ``.xy.RF_frequency``,
        and optionally ``.freq_vs_flux_01_quad_term`` / ``.extras``).
    detuning_hz :
        Target |Δf| below idle (Hz).
    flux_branch :
        Preferred parabola side (``\"left\"`` / ``\"right\"``).
    freq_to_flux_source :
        ``\"auto\"`` (Ramsey → spectroscopy → quad_term) or a forced source.

    Returns
    -------
    ResolvedFluxAmps
        Amplitudes, source labels, the per-qubit chosen curves, and the
        effective branch (last qubit's branch is stored on the dataclass — fine
        for single-qubit / shared branch runs).
    """
    amplitudes: List[float] = []
    sources: List[str] = []
    curves: Dict[str, FreqFluxCurve] = {}
    effective_branch: Branch = flux_branch

    for q in qubits:
        amp: Optional[float] = None
        label: Optional[str] = None
        used_branch: Branch = flux_branch
        idle = q.xy.RF_frequency

        selected = resolve_freq_flux_curve(q, freq_to_flux_source)
        curves[q.name] = selected

        if selected.is_measured:
            for br in (flux_branch, _OPPOSITE[flux_branch]):
                amp = flux_amp_from_curve(detuning_hz, idle, selected.curve, br)
                if amp is not None:
                    used_branch = br
                    if br != flux_branch:
                        warnings.warn(
                            f"{q.name}: target detuning not found on {flux_branch} branch of "
                            f"{selected.label}, trying {br}"
                        )
                        label = f"{selected.label} ({br}, fallback)"
                    else:
                        label = f"{selected.label} ({br})"
                    break
            if amp is None:
                warnings.warn(
                    f"{q.name}: target detuning {detuning_hz / 1e6:.1f} MHz is not reachable on "
                    f"either branch of {selected.label}; falling back to freq_vs_flux_01_quad_term."
                )

        if amp is None:
            qt = selected.quad_term if selected.quad_term is not None else getattr(q, "freq_vs_flux_01_quad_term", None)
            if qt is not None and qt != 0 and np.isfinite(qt):
                sign = 1.0 if flux_branch == "right" else -1.0
                amp = sign * float(np.sqrt(abs(detuning_hz) / abs(qt)))
                label = f"quad_term={qt:.3e}"
                used_branch = flux_branch
                if selected.kind != "quad_term":
                    curves[q.name] = FreqFluxCurve(
                        kind="quad_term", label=f"quad_term={float(qt):.3e}", quad_term=float(qt)
                    )

        if amp is None:
            raise ValueError(
                f"Cannot derive flux_amp for {q.name}: no usable freq-vs-flux curve "
                f"(freq_to_flux_source='{freq_to_flux_source}') and freq_vs_flux_01_quad_term "
                f"is missing or zero."
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
        curves=curves,
    )
