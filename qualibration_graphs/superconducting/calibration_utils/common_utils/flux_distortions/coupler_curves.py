"""Coupler-flux freq↔flux helpers for flux-distortion calibrations (21a/c and similar).

Two jobs share the same loaded curves:

1. **Pre-run** — pick a coupler flux-pulse amplitude for a **signed** detuning from
   the decouple point (``coupler_flux_amp_from_curve`` / ``resolve_coupler_flux_amplitudes``).
2. **Post-run** — invert measured absolute qubit frequency vs time into coupler flux
   (``frequency_to_coupler_flux``).

Curves come from prior calibrations; run IDs are read from ``qubit.extras`` on the
measured qubit — never entered by hand:

* **03c** qubit spectroscopy vs coupler flux → ``ds_fit.peak_freq`` (relative to RF),
  run ID in ``extras['{coupler.name}_spectroscopy_dispersion_load_id']``
* **09b** Ramsey vs coupler flux → absolute qubit frequency vs coupler flux,
  run ID in ``extras['{coupler.name}_ramsey_dispersion_load_id']``

``resolve_coupler_freq_flux_curve`` is the single source-selection point (mirrors
``resolve_freq_flux_curve``). There is no ``quad_term`` fallback; use
``coupler_flux_amplitude_in_v`` on the node when no curve is available.

Reference flux is the coupler **decouple point** (0 V in the played-relative frame),
not the qubit idle sweetspot.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qualibrate import QualibrationNode
from quam_builder.architecture.superconducting.components.tunable_coupler import TunableCoupler
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .curves import FreqFluxCurve, MeasuredCurve, _resolve_measured_curve, extras_run_id
from .node_storage import read_node_data_dict

LogCallable = Callable[[str], None]

CouplerFreqFluxSource = Literal["auto", "spectroscopy", "ramsey"]
CouplerCurveKind = Literal["spectroscopy", "ramsey"]

#: Priority order for ``source="auto"`` (spectroscopy first for coupler cryoscope nodes).
COUPLER_AUTO_SOURCE_ORDER: Tuple[CouplerCurveKind, ...] = ("spectroscopy", "ramsey")

# Backward-compatible alias — same dataclass as qubit side (``quad_term`` unused).
CouplerFreqFluxCurve = FreqFluxCurve


def coupler_extras_key(coupler: TunableCoupler | str, kind: CouplerCurveKind) -> str:
    """Extras field name for a coupler dispersion run ID on the measured qubit."""
    name = coupler.name if hasattr(coupler, "name") else str(coupler)
    suffix = "spectroscopy_dispersion_load_id" if kind == "spectroscopy" else "ramsey_dispersion_load_id"
    return f"{name}_{suffix}"


# ---------------------------------------------------------------------------
# Loaders (03c / 09b)
# ---------------------------------------------------------------------------


def _resolve_pair_coord_key(
    qubit: AnyTransmon,
    coupler: TunableCoupler,
    node: QualibrationNode[Any, Any],
    ds_qubit_values,
) -> Optional[str]:
    qvals = [str(v) for v in np.atleast_1d(ds_qubit_values)]
    if qubit.name in qvals:
        return qubit.name
    for pair_name, pair in node.machine.qubit_pairs.items():
        if pair.coupler.name == coupler.name and str(pair_name) in qvals:
            return str(pair_name)
    return None


def load_coupler_spectroscopy_curve(
    qubit: AnyTransmon,
    coupler: TunableCoupler,
    node: QualibrationNode[Any, Any],
    run_id: Optional[int] = None,
    *,
    log_callable: Optional[LogCallable] = None,
) -> Optional[MeasuredCurve]:
    """Load qubit-freq vs coupler-flux from a 03c run for one pair."""
    rid = (
        int(run_id)
        if run_id is not None
        else extras_run_id(qubit, coupler_extras_key(coupler, "spectroscopy"), log_callable=log_callable)
    )
    if rid is None:
        return None

    qubit_name = qubit.name
    qubit_rf = float(qubit.xy.RF_frequency)
    if log_callable is not None:
        log_callable(f"Loading coupler spectroscopy #{rid} for {qubit_name} / {coupler.name}")
    try:
        ds_raw = read_node_data_dict(rid)["ds_raw"]
        key = _resolve_pair_coord_key(qubit, coupler, node, ds_raw.qubit.values)
        if key is None and "measured_qubit_name" in ds_raw.coords:
            qvals = [str(v) for v in np.atleast_1d(ds_raw.qubit.values)]
            matches = [
                qv
                for qv, m in zip(qvals, np.atleast_1d(ds_raw.measured_qubit_name.values))
                if str(m) == qubit_name
            ]
            if len(matches) == 1:
                key = matches[0]
        if key is None:
            if log_callable is not None:
                log_callable(
                    f"{qubit_name}/{coupler.name}: could not resolve coord in run #{rid}; "
                    f"skipping spectroscopy path"
                )
            return None

        ds_fit = read_node_data_dict(rid).get("ds_fit")
        if ds_fit is None or "peak_freq" not in getattr(ds_fit, "data_vars", {}):
            if log_callable is not None:
                log_callable(f"run #{rid} has no ds_fit.peak_freq for {qubit_name}/{coupler.name}")
            return None

        flux = np.asarray(ds_fit.flux_bias.values, dtype=float)
        peak = np.asarray(ds_fit.peak_freq.sel(qubit=key).values, dtype=float)
        freq = qubit_rf + peak
        mask = np.isfinite(flux) & np.isfinite(freq)
        if mask.sum() < 2:
            if log_callable is not None:
                log_callable(f"Too few finite spectroscopy points for {qubit_name} in run #{rid}")
            return None
        flux_m, freq_m = flux[mask], freq[mask]
        order = np.argsort(flux_m)
        if log_callable is not None:
            log_callable(
                f"Loaded coupler spectroscopy for {qubit_name} (coord '{key}') from run #{rid}: "
                f"{mask.sum()} pts"
            )
        return flux_m[order], freq_m[order]
    except Exception as e:
        if log_callable is not None:
            log_callable(f"Failed to load coupler spectroscopy for {qubit_name} from run #{rid}: {e}")
        return None


def load_coupler_ramsey_curve(
    qubit: AnyTransmon,
    coupler: TunableCoupler,
    node: QualibrationNode[Any, Any],
    run_id: Optional[int] = None,
    *,
    frequency_var: str = "abs_peak_frequency",
    log_callable: Optional[LogCallable] = None,
) -> Optional[MeasuredCurve]:
    """Load qubit-freq vs coupler-flux from a 09b run for one pair."""
    rid = (
        int(run_id)
        if run_id is not None
        else extras_run_id(qubit, coupler_extras_key(coupler, "ramsey"), log_callable=log_callable)
    )
    if rid is None:
        return None

    if log_callable is not None:
        log_callable(f"Loading coupler Ramsey #{rid} for {qubit.name} / {coupler.name}")
    try:
        ds_fit = read_node_data_dict(rid)["ds_fit"]

        if "qubit_frequency" in ds_fit.data_vars and "coupler_flux" in ds_fit.dims:
            flux_bias_rel = ds_fit.coupler_flux.values
            if "qubit_pair" in ds_fit.dims:
                qp_names = [str(qp) for qp in ds_fit.qubit_pair.values]
                pair_key = None
                for pair_name, pair in node.machine.qubit_pairs.items():
                    if pair.coupler.name == coupler.name and str(pair_name) in qp_names:
                        pair_key = str(pair_name)
                        break
                if pair_key is None:
                    if log_callable is not None:
                        log_callable(f"No qubit_pair match for {qubit.name} / {coupler.name} in run #{rid}")
                    return None
                frequency = ds_fit["qubit_frequency"].sel(qubit_pair=pair_key).values
            else:
                frequency = ds_fit["qubit_frequency"].values
            return flux_bias_rel, frequency

        flux_bias_rel = ds_fit["coupler_flux"].values
        pair_key = None
        for pair_name, pair in node.machine.qubit_pairs.items():
            if pair.coupler.name == coupler.name:
                pair_key = str(pair_name)
                break
        if pair_key is None or frequency_var not in ds_fit.data_vars:
            if log_callable is not None:
                log_callable(
                    f"Cannot find '{frequency_var}' for {qubit.name} / {coupler.name} in run #{rid}"
                )
            return None
        frequency = ds_fit[frequency_var].sel(qubit_pair=pair_key).values
        return flux_bias_rel, frequency
    except Exception as e:
        if log_callable is not None:
            log_callable(f"Failed to load coupler Ramsey for {qubit.name} from run #{rid}: {e}")
        return None


# ---------------------------------------------------------------------------
# Source selection (mirrors ``resolve_freq_flux_curve``)
# ---------------------------------------------------------------------------


def resolve_coupler_freq_flux_curve(
    qubit: AnyTransmon,
    coupler: TunableCoupler,
    node: QualibrationNode[Any, Any],
    source: CouplerFreqFluxSource = "auto",
    *,
    log_callable: Optional[LogCallable] = None,
) -> FreqFluxCurve:
    """Pick the coupler freq↔flux relation for one pair."""
    measured = _resolve_measured_curve(
        source,
        order=COUPLER_AUTO_SOURCE_ORDER,
        loaders={
            "spectroscopy": lambda: load_coupler_spectroscopy_curve(
                qubit, coupler, node, log_callable=log_callable
            ),
            "ramsey": lambda: load_coupler_ramsey_curve(qubit, coupler, node, log_callable=log_callable),
        },
        run_id_for_kind=lambda kind: extras_run_id(
            qubit, coupler_extras_key(coupler, kind), log_callable=log_callable  # type: ignore[arg-type]
        ),
        label_for_kind=lambda kind, rid: (
            f"{'spectroscopy' if kind == 'spectroscopy' else 'Ramsey'} #{rid}"
            if rid is not None
            else ("spectroscopy" if kind == "spectroscopy" else "Ramsey")
        ),
        log_forced_miss=(
            lambda kind: log_callable(
                f"{qubit.name}/{coupler.name}: freq_to_flux_source='{kind}' was requested but no usable "
                f"curve could be loaded (extras['{coupler_extras_key(coupler, kind)}'] missing or unreadable)."
            )
            if log_callable is not None
            else None
        ),
    )
    if measured is not None:
        return measured

    return FreqFluxCurve(kind="none", label="unavailable")


# ---------------------------------------------------------------------------
# Mapping: signed detuning → amp (pre-run) and freq → Φ (post-run)
# ---------------------------------------------------------------------------


def coupler_flux_amp_from_curve(
    detuning_hz: float,
    curve: MeasuredCurve,
    *,
    decouple_offset: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Invert a dispersion curve to a coupler flux amplitude for a signed detuning.

    Returns ``(playable_flux_V, freq_at_decouple_Hz)``. When multiple flux values
    reach the target frequency, picks the crossing nearest the decouple point.
    """
    curve_flux, curve_freq = curve
    if len(curve_flux) < 2:
        return None

    decouple_idx = int(np.argmin(np.abs(curve_flux - decouple_offset)))
    decouple_flux_on_curve = float(curve_flux[decouple_idx])
    if abs(decouple_flux_on_curve - decouple_offset) > 0.01:
        warnings.warn(
            f"The decouple-point reference ({decouple_offset:.4f} V) is not well covered "
            f"by the curve range [{curve_flux.min():.4f}, {curve_flux.max():.4f}] V "
            f"(nearest sample={decouple_flux_on_curve:.4f} V)."
        )
    freq_at_decouple = float(curve_freq[decouple_idx])
    target_freq = freq_at_decouple + detuning_hz

    diff = curve_freq - target_freq
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        warnings.warn(
            f"Target frequency {target_freq / 1e9:.6f} GHz is outside the dispersion curve "
            f"range [{curve_freq.min() / 1e9:.6f}, {curve_freq.max() / 1e9:.6f}] GHz."
        )
        return None

    crossing_fluxes = []
    for idx in sign_changes:
        f1, f2 = curve_freq[idx], curve_freq[idx + 1]
        x1, x2 = curve_flux[idx], curve_flux[idx + 1]
        frac = (target_freq - f1) / (f2 - f1) if abs(f2 - f1) > 0 else 0.0
        crossing_fluxes.append(float(x1 + frac * (x2 - x1)))

    best = int(np.argmin([abs(f - decouple_offset) for f in crossing_fluxes]))
    return crossing_fluxes[best], freq_at_decouple


def frequency_to_coupler_flux(
    measured_abs_freq: NDArray[np.floating],
    curve: MeasuredCurve,
    n_flux_fine: int = 1000,
) -> NDArray[np.floating]:
    """Map measured absolute qubit frequency to coupler flux on ``curve``."""
    flux_bias, abs_peak_frequency = curve
    flux_min, flux_max = flux_bias.min(), flux_bias.max()
    idx_s = np.argsort(flux_bias)
    flux_s = flux_bias[idx_s]
    freq_s = abs_peak_frequency[idx_s]

    flux_fine = np.linspace(flux_min, flux_max, n_flux_fine)
    freq_fine = np.interp(flux_fine, flux_s, freq_s)

    idx_inv = np.argsort(freq_fine)
    freq_sorted = freq_fine[idx_inv]
    flux_sorted = flux_fine[idx_inv]
    freq_unique, idx_unique = np.unique(freq_sorted, return_index=True)
    flux_unique = flux_sorted[idx_unique]

    measured_flat = np.asarray(measured_abs_freq, dtype=float).ravel()
    out = np.interp(
        measured_flat,
        freq_unique,
        flux_unique,
        left=flux_min,
        right=flux_max,
    )
    out = np.clip(out, flux_min, flux_max)
    return out.reshape(np.shape(measured_abs_freq))


# ---------------------------------------------------------------------------
# Pre-run batch (mirrors ``resolve_flux_amplitudes``)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCouplerFluxAmps:
    """Result of ``resolve_coupler_flux_amplitudes`` for a qubit-pair batch."""

    amplitudes: List[float]
    sources: List[str]
    freq_at_decouple: List[Optional[float]]
    curves: Dict[str, FreqFluxCurve] = field(default_factory=dict)


def resolve_coupler_flux_amplitudes(
    qubit_pairs: Iterable[Any],
    *,
    measure_qubit: Literal["control", "target"],
    detuning_hz: float,
    freq_to_flux_source: CouplerFreqFluxSource = "auto",
    fallback_amplitude_v: Optional[float] = None,
    node: Optional[QualibrationNode[Any, Any]] = None,
    log_callable: Optional[LogCallable] = None,
) -> ResolvedCouplerFluxAmps:
    """Derive per-pair coupler flux pulse amplitudes for a signed detuning."""
    amplitudes: List[float] = []
    sources: List[str] = []
    freq_at_decouple_list: List[Optional[float]] = []
    curves: Dict[str, FreqFluxCurve] = {}

    for qp in qubit_pairs:
        qubit = qp.qubit_control if measure_qubit == "control" else qp.qubit_target
        amp: Optional[float] = None
        f_dec: Optional[float] = None
        label = "unavailable"

        if node is not None:
            selected = resolve_coupler_freq_flux_curve(
                qubit, qp.coupler, node, freq_to_flux_source, log_callable=log_callable
            )
            curves[qp.name] = selected
            if selected.is_measured:
                result = coupler_flux_amp_from_curve(detuning_hz, selected.curve)
                if result is not None:
                    amp, f_dec = result
                    label = selected.label

        if amp is None and fallback_amplitude_v is not None and fallback_amplitude_v != 0:
            amp = float(fallback_amplitude_v)
            label = f"coupler_flux_amplitude={amp:.4f} V (user input)"

        if amp is None:
            raise ValueError(
                f"Cannot derive coupler flux for {qp.name} / {qp.coupler.name}. "
                f"detuning={detuning_hz / 1e6:+.2f} MHz. "
                f"Run 03c / 09b with save_load_id=True or set coupler_flux_amplitude_in_v."
            )

        if abs(amp) > 0.5:
            msg = (
                f"{qp.name}: derived coupler_flux={amp:.4f} V exceeds 0.5 V. "
                f"Verify detuning is correct."
            )
            if log_callable is not None:
                log_callable(msg)
            else:
                warnings.warn(msg)

        amplitudes.append(amp)
        sources.append(label)
        freq_at_decouple_list.append(f_dec)

    return ResolvedCouplerFluxAmps(
        amplitudes=amplitudes,
        sources=sources,
        freq_at_decouple=freq_at_decouple_list,
        curves=curves,
    )

