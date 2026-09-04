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

Branch handling: the qubit is assumed to sit at its flux sweetspot, where
``f(Φ)`` is symmetric about idle and both flux directions detune downwards by the
same amount. Which side of the parabola is used is therefore not a user choice
— these helpers pick whichever side reaches (pre-run) or covers (post-run) the
target and return the **magnitude** of the offset from idle, |ΔΦ|. That is all
the IIR taps ``A_i = a_i / a_dc`` depend on: a global sign on the step response
cancels in the ratio. Offsets are relative to idle, not absolute DAC volts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .node_storage import read_node_data_dict

LogCallable = Callable[[str], None]

MeasuredCurve = Tuple[NDArray[np.floating], NDArray[np.floating]]
FreqFluxKind = Literal["ramsey", "spectroscopy", "quad_term", "none"]

#: Side of the parabola, for the low-level curve helpers only. Not a node
#: parameter: at the sweetspot both sides are equivalent (see module docstring).
Branch = Literal["left", "right"]

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


def extras_run_id(
    qubit: AnyTransmon,
    key: str,
    *,
    log_callable: Optional[LogCallable] = None,
) -> Optional[int]:
    """Return the Qualibrate snapshot index stored on a qubit in ``state.json``.

    Prior calibrations that produce a freq-vs-flux curve (09a Ramsey vs Z, 03b
    spectroscopy vs Z) can record *which run* produced that curve by writing the
    Qualibrate snapshot index into ``qubit.extras`` when ``save_load_id=True``.
    Downstream nodes (17a, 17b, …) call this helper to recover that index and
    load the saved curve — the user never types a run ID into the GUI.

    Typical keys (see module constants above):

    * ``RAMSEY_EXTRAS_KEY`` — written by 09a
    * ``SPECTROSCOPY_EXTRAS_KEY`` — written by 03b

    Parameters
    ----------
    qubit :
        QUAM qubit object; must expose ``.name`` and optionally ``.extras``.
    key :
        Name of the extras field that holds the snapshot index (e.g.
        ``"ramsey_vs_flux_calibration_load_id"``).

    Returns
    -------
    int or None
        Parsed snapshot index, or ``None`` if the key is missing, ``extras`` is
        empty, or the stored value is not an integer.
    """
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
        if log_callable is not None:
            log_callable(f"{getattr(qubit, 'name', '?')}: extras['{key}']={value!r} is not a run ID")
        return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_spectroscopy_curve(
    qubit: AnyTransmon,
    run_id: Optional[int] = None,
    *,
    log_callable: Optional[LogCallable] = None,
) -> Optional[MeasuredCurve]:
    """Load freq-vs-Z-flux from a 03b spectroscopy-vs-flux run for ``qubit``.

    Uses ``ds_fit.peak_freq`` (fit peak relative to the RF LO / drive) and adds
    ``qubit.xy.RF_frequency`` so the returned frequency axis is absolute Hz.

    The run ID comes from ``run_id`` if given, else
    ``qubit.extras['qubit_spectroscopy_vs_flux_load_id']`` (written by 03b when
    ``save_load_id=True``) — same pattern as ``load_ramsey_curve``.

    Parameters
    ----------
    qubit :
        QUAM qubit (needs ``.name``, ``.xy.RF_frequency``, optionally ``.extras``).
    run_id :
        Optional Qualibrate snapshot / run ID override. When omitted, the extras
        key is used.

    Returns
    -------
    (flux_V, freq_Hz) or None
        Flux-sorted 1-D arrays. ``None`` if no run ID, the run is missing / has
        no ``peak_freq``, or has fewer than two finite points.
    """
    rid = int(run_id) if run_id is not None else extras_run_id(qubit, SPECTROSCOPY_EXTRAS_KEY, log_callable=log_callable)
    if rid is None:
        return None

    qubit_name = qubit.name
    qubit_rf_freq = float(qubit.xy.RF_frequency)
    try:
        data = read_node_data_dict(rid)
        ds_fit = data.get("ds_fit")
        if ds_fit is None or "peak_freq" not in ds_fit:
            if log_callable is not None:
                log_callable(
                    f"run #{rid} has no ds_fit.peak_freq for {qubit_name}; cannot load spectroscopy curve"
                )
            return None
        flux = np.asarray(ds_fit.flux_bias.values, dtype=float)
        peak = np.asarray(ds_fit.peak_freq.sel(qubit=qubit_name).values, dtype=float)
        freq = qubit_rf_freq + peak
        mask = np.isfinite(flux) & np.isfinite(freq)
        if mask.sum() < 2:
            if log_callable is not None:
                log_callable(f"Too few finite peak_freq points for {qubit_name} in run #{rid}")
            return None
        flux_m, freq_m = flux[mask], freq[mask]
        order = np.argsort(flux_m)
        flux_m, freq_m = flux_m[order], freq_m[order]
        if log_callable is not None:
            log_callable(
                f"Loaded spectroscopy curve for {qubit_name} from run #{rid} "
                f"(ds_fit.peak_freq): {len(flux_m)} pts, flux=[{flux_m[0]:.4f}, {flux_m[-1]:.4f}] V"
            )
        return flux_m, freq_m
    except Exception as e:
        if log_callable is not None:
            log_callable(f"Failed to load spectroscopy curve for {qubit_name} from run #{rid}: {e}")
        return None


def load_ramsey_curve(
    qubit: AnyTransmon,
    run_id: Optional[int] = None,
    *,
    log_callable: Optional[LogCallable] = None,
) -> Optional[MeasuredCurve]:
    """Load freq-vs-Z-flux from a 09a Ramsey-vs-flux run for ``qubit``.

    The run ID comes from ``run_id`` if given, else
    ``qubit.extras['ramsey_vs_flux_calibration_load_id']`` (written by 09a when
    ``save_load_id=True``).

    Preference order for the frequency axis on ``ds_fit``:

    1. ``f_qubit_vs_flux`` (GHz) → absolute Hz.
    2. Else ``unfolded_frequency`` (GHz), converted to absolute qubit frequency
       via ``RF + artificial_detuning − ramsey_freq``.

    Parameters
    ----------
    qubit :
        QUAM qubit (needs ``.name``, ``.xy.RF_frequency``, optionally ``.extras``).
    run_id :
        Optional Qualibrate snapshot / run ID override. When omitted, the extras
        key is used.

    Returns
    -------
    (flux_V, freq_Hz) or None
        Finite points only. ``None`` if no run ID, all-NaN, or load error.
    """
    rid = int(run_id) if run_id is not None else extras_run_id(qubit, RAMSEY_EXTRAS_KEY, log_callable=log_callable)
    if rid is None:
        return None

    qubit_name = qubit.name
    qubit_rf_freq = float(qubit.xy.RF_frequency)
    if log_callable is not None:
        log_callable(f"Loading Ramsey #{rid} for {qubit_name}")
    try:
        data = read_node_data_dict(rid)
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
            if log_callable is not None:
                log_callable(f"All NaN in Ramsey curve for {qubit_name} from run #{rid}")
            return None
        if log_callable is not None:
            log_callable(
                f"{qubit_name}: Ramsey #{rid} — {mask.sum()} pts, "
                f"flux [{flux_bias[mask][0]:.3f}, {flux_bias[mask][-1]:.3f}] V, "
                f"freq [{freq_hz[mask].min() / 1e9:.3f}, {freq_hz[mask].max() / 1e9:.3f}] GHz"
            )
        return flux_bias[mask], freq_hz[mask]
    except Exception as e:
        if log_callable is not None:
            log_callable(f"Failed to load Ramsey curve for {qubit_name} from run #{rid}: {e}")
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

    kind: FreqFluxKind
    label: str
    curve: Optional[MeasuredCurve] = None
    run_id: Optional[int] = None
    quad_term: Optional[float] = None

    @property
    def is_measured(self) -> bool:
        """True when a measured freq-vs-flux ``curve`` is available."""
        return self.curve is not None


def _source_run_id(qubit: AnyTransmon, kind: str) -> Optional[int]:
    """Extras run ID backing source ``kind`` for ``qubit``."""
    if kind == "ramsey":
        return extras_run_id(qubit, RAMSEY_EXTRAS_KEY)
    if kind == "spectroscopy":
        return extras_run_id(qubit, SPECTROSCOPY_EXTRAS_KEY)
    return None


def resolve_freq_flux_curve(
    qubit: AnyTransmon,
    source: FreqFluxSource = "auto",
    *,
    log_callable: Optional[LogCallable] = None,
) -> FreqFluxCurve:
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
        if kind == "ramsey":
            curve = load_ramsey_curve(qubit, log_callable=log_callable)
        elif kind == "spectroscopy":
            curve = load_spectroscopy_curve(qubit, log_callable=log_callable)
        else:
            curve = None
        if curve is not None:
            rid = _source_run_id(qubit, kind)
            pretty = "Ramsey" if kind == "ramsey" else "spectroscopy"
            return FreqFluxCurve(kind=kind, label=f"{pretty} #{rid}", curve=curve, run_id=rid)
        if source != "auto":
            extras_key = RAMSEY_EXTRAS_KEY if kind == "ramsey" else SPECTROSCOPY_EXTRAS_KEY
            if log_callable is not None:
                log_callable(
                    f"{qubit.name}: freq_to_flux_source='{kind}' was requested but no usable curve "
                    f"could be loaded (extras['{extras_key}'] missing or unreadable). Falling back to "
                    f"freq_vs_flux_01_quad_term — re-run the source calibration with save_load_id=True."
                )

    qt = getattr(qubit, "freq_vs_flux_01_quad_term", None)
    if qt is not None and qt != 0 and np.isfinite(qt):
        return FreqFluxCurve(kind="quad_term", label=f"quad_term={float(qt):.3e}", quad_term=float(qt))

    return FreqFluxCurve(kind="none", label="unavailable")


def resolve_freq_flux_curves(
    qubits: Iterable[AnyTransmon],
    source: FreqFluxSource = "auto",
    *,
    log_callable: Optional[LogCallable] = None,
) -> Dict[str, FreqFluxCurve]:
    """``resolve_freq_flux_curve`` for a qubit batch, keyed by qubit name."""
    return {q.name: resolve_freq_flux_curve(q, source, log_callable=log_callable) for q in qubits}


# ---------------------------------------------------------------------------
# Mapping: detuning → amp (pre-run) and freq → ΔΦ (post-run)
# ---------------------------------------------------------------------------


def flux_amp_from_curve(
    detuning_hz: float,
    idle_freq_hz: float,
    curve: MeasuredCurve,
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


def _strictly_increasing_in_freq(
    b_flux: NDArray[np.floating],
    b_freq: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sort a branch by frequency and drop non-increasing repeats.

    ``CubicSpline`` requires a strictly increasing abscissa. Measured curves can
    repeat or invert a frequency (noise on adjacent flux points, or a sample
    straddling the vertex), so keep the first point of each such group.
    """
    order = np.argsort(b_freq)
    f_sorted, x_sorted = b_freq[order], b_flux[order]
    keep = np.ones(len(f_sorted), dtype=bool)
    last = -np.inf
    for i, f in enumerate(f_sorted):
        if f > last:
            last = f
        else:
            keep[i] = False
    return x_sorted[keep], f_sorted[keep]


def _pick_inversion_branch(
    curve_flux: NDArray[np.floating],
    curve_freq: NDArray[np.floating],
    idle_freq: float,
    measured_abs_freq: NDArray[np.floating],
) -> Optional[Tuple[NDArray[np.floating], NDArray[np.floating], float]]:
    """Return the branch of ``curve`` that best covers the measured frequencies.

    At the sweetspot both sides of the parabola are physically equivalent, so the
    only thing that distinguishes them is sampling: pick the side that brackets
    more of ``measured_abs_freq``, breaking ties on the number of points.

    The split is by **index** around the idle sample rather than by flux value,
    so a sample sitting just off the vertex cannot land on both sides and make
    the branch non-monotonic in frequency.

    Returns
    -------
    (branch_flux, branch_freq, idle_flux) or None
        Arrays are strictly increasing in frequency, ready for a cubic spline.
        ``None`` when neither side keeps the >= 4 points a spline needs.
    """
    order = np.argsort(np.asarray(curve_flux, dtype=float))
    flux = np.asarray(curve_flux, dtype=float)[order]
    freq = np.asarray(curve_freq, dtype=float)[order]

    idle_idx = int(np.argmin(np.abs(freq - idle_freq)))
    idle_flux = float(flux[idle_idx])

    measured = np.asarray(measured_abs_freq, dtype=float).ravel()
    finite = measured[np.isfinite(measured)]
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    best_score = (-1, -1)

    for b_flux, b_freq in (
        (flux[idle_idx:], freq[idle_idx:]),
        (flux[: idle_idx + 1], freq[: idle_idx + 1]),
    ):
        b_flux, b_freq = _strictly_increasing_in_freq(b_flux, b_freq)
        if len(b_flux) < 4:
            continue
        covered = int(np.sum((finite >= b_freq.min()) & (finite <= b_freq.max()))) if finite.size else 0
        score = (covered, len(b_flux))
        if score > best_score:
            best, best_score = (b_flux, b_freq, idle_flux), score

    return best


def frequency_to_flux_deviation(
    measured_abs_freq: NDArray[np.floating],
    curve_flux: NDArray[np.floating],
    curve_freq: NDArray[np.floating],
    idle_freq: float,
) -> NDArray[np.floating]:
    """Map measured absolute qubit frequency → flux offset magnitude from idle.

    Used in analysis after center frequencies vs time are extracted: invert the
    same freq↔flux curve so the cryoscope / π-vs-flux trace becomes a flux step
    response |ΔΦ|(t).

    Method: pick the better-sampled branch with ``_pick_inversion_branch`` (both
    are equivalent at the sweetspot), fit a ``scipy.interpolate.CubicSpline`` of
    flux(freq) on it, evaluate at ``measured_abs_freq``, and take
    ``|flux - idle_flux|``.

    Points landing further than 2x the branch flux span from idle are set to NaN
    (well outside where the curve constrains anything).

    Parameters
    ----------
    measured_abs_freq :
        Absolute qubit frequency (Hz), any shape; returned array matches shape.
    curve_flux, curve_freq :
        Dispersion curve arrays (same convention as the loaders).
    idle_freq :
        Idle frequency (Hz) used to locate idle flux on the curve.

    Returns
    -------
    np.ndarray
        |ΔΦ| in volts, same shape as ``measured_abs_freq``. All-NaN if neither
        branch has at least 4 usable points.
    """
    from scipy.interpolate import CubicSpline

    branch = _pick_inversion_branch(curve_flux, curve_freq, idle_freq, measured_abs_freq)
    if branch is None:
        return np.full_like(np.asarray(measured_abs_freq, dtype=float), np.nan)
    b_flux, b_freq, idle_flux = branch

    cs_inv = CubicSpline(b_freq, b_flux, extrapolate=True)
    measured_flat = np.asarray(measured_abs_freq, dtype=float).ravel()
    result = np.abs(cs_inv(measured_flat) - idle_flux)

    flux_range = float(np.abs(b_flux - idle_flux).max())
    result[result > 2.0 * flux_range] = np.nan
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
        Z-pulse amplitude magnitudes (V) aligned with the input qubit list.
        Positive by construction: at the sweetspot either flux direction reaches
        the target detuning, and the sign never reaches the IIR taps.
    sources :
        Human-readable origin per qubit, e.g. ``"Ramsey #42"`` or
        ``"quad_term=1.2e9"``.
    curves :
        The ``FreqFluxCurve`` chosen per qubit, keyed by qubit name — the same
        selection analysis will make, so it can be reported to the user.
    """

    amplitudes: List[float]
    sources: List[str]
    curves: Dict[str, FreqFluxCurve] = field(default_factory=dict)


def resolve_flux_amplitudes(
    qubits: Iterable[AnyTransmon],
    *,
    detuning_hz: float,
    freq_to_flux_source: FreqFluxSource = "auto",
    log_callable: Optional[LogCallable] = None,
) -> ResolvedFluxAmps:
    """Derive per-qubit Z-pulse amplitudes for a target detuning.

    Intended for node setup (e.g. 17a ``create_qua_program``) before the QUA
    program is built. The freq↔flux relation is chosen once per qubit by
    ``resolve_freq_flux_curve`` — the same call analysis makes — so the amplitude
    and the later flux inversion always use the same relation.

    Given that relation:

    * **Measured curve** (Ramsey / spectroscopy) → ``flux_amp_from_curve`` over
      the whole curve, taking the smallest-|ΔΦ| crossing on either side.
    * **Quadratic** → ``sqrt(|detuning| / |quad_term|)``.

    Amplitudes are magnitudes: the qubit is assumed to be at its sweetspot, so
    both flux directions detune downwards equally and the sign is irrelevant to
    the fitted taps.

    Raises ``ValueError`` if no relation is usable for a qubit. Warns if
    ``amp > 0.5`` V.

    Parameters
    ----------
    qubits :
        Iterable of QUAM qubit objects (need ``.name``, ``.xy.RF_frequency``,
        and optionally ``.freq_vs_flux_01_quad_term`` / ``.extras``).
    detuning_hz :
        Target |Δf| below idle (Hz).
    freq_to_flux_source :
        ``"auto"`` (Ramsey → spectroscopy → quad_term) or a forced source.

    Returns
    -------
    ResolvedFluxAmps
        Amplitudes, source labels, and the per-qubit chosen curves.
    """
    amplitudes: List[float] = []
    sources: List[str] = []
    curves: Dict[str, FreqFluxCurve] = {}

    for q in qubits:
        amp: Optional[float] = None
        label: Optional[str] = None
        idle = q.xy.RF_frequency

        selected = resolve_freq_flux_curve(q, freq_to_flux_source, log_callable=log_callable)
        curves[q.name] = selected

        if selected.is_measured:
            # branch=None: smallest |ΔΦ| crossing on either side of idle.
            amp = flux_amp_from_curve(detuning_hz, idle, selected.curve, None)
            if amp is None:
                if log_callable is not None:
                    log_callable(
                        f"{q.name}: target detuning {detuning_hz / 1e6:.1f} MHz is not reachable on "
                        f"{selected.label}; falling back to freq_vs_flux_01_quad_term."
                    )
            else:
                label = selected.label

        if amp is None:
            qt = selected.quad_term if selected.quad_term is not None else getattr(q, "freq_vs_flux_01_quad_term", None)
            if qt is not None and qt != 0 and np.isfinite(qt):
                amp = float(np.sqrt(abs(detuning_hz) / abs(qt)))
                label = f"quad_term={qt:.3e}"
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

        if amp > 0.5:
            if log_callable is not None:
                log_callable(
                    f"{q.name}: derived flux_amp={amp:.4f} V exceeds 0.5 V. Verify detuning_in_mhz "
                    f"is correct — note the OPX output range must also accommodate the standing "
                    f"flux offset, so usable headroom is less than the derived amplitude suggests."
                )

        amplitudes.append(float(amp))
        sources.append(label or "unknown")

    return ResolvedFluxAmps(amplitudes=amplitudes, sources=sources, curves=curves)
