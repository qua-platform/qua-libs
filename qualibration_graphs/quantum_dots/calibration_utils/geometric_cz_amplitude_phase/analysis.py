"""Analysis for the 2-D CZ amplitude × analysis-phase calibration.

For each exchange amplitude V and analysis phase θ the parity signal is
acquired for both control-qubit states.  The parity difference is:

    D(V, θ) = S(ctrl=|1⟩, V, θ) − S(ctrl=|0⟩, V, θ)
             = −cos(J(V)·t/2) · cos θ   (simplified exchange model)

The phase average ⟨D⟩_θ = 0 for any uniform sampling over [0, 2π), so it
carries no information about V.

The mean absolute parity difference is:

    mean_abs_diff(V) = ⟨|D(V, θ)|⟩_θ  ∝  |cos(J(V)·t/2)|

This is MINIMISED at the CZ operating point where J(V*)·t/2 = π/2
(conditional phase δ = π), so:

    V* = argmin_V mean_abs_diff(V)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class AmplitudePhaseFitResult:
    """Result of the 2-D amplitude-phase CZ calibration for one qubit pair."""

    exchange_duration: float = 0.0
    """Fixed exchange pulse duration used (ns).  0 when per-amplitude durations are used."""
    optimal_amplitude: float = 0.0
    """Amplitude V* where ⟨|D|⟩_θ is minimised (V)."""
    optimal_duration: int = 0
    """Exchange duration at V* (ns, multiple of 4).  Equals exchange_duration
    in fixed-duration mode, or the model-derived value in per-amplitude mode."""
    min_mean_abs_diff: float = float("nan")
    """Value of ⟨|D|⟩_θ at V* (approaches 0 at the ideal CZ point)."""
    success: bool = False


def analyse_amplitude_phase(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    exchange_duration: float = 0.0,
    duration_array: np.ndarray | None = None,
    analysis_signal: str = "E_p2_given_p1_0",
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Find the CZ amplitude that minimises the mean absolute parity difference.

    Args:
        ds_raw: Dataset with dims (control_state, exchange_amplitude, analysis_phase).
            Must contain ``{analysis_signal}_{pair.name}`` for each pair.
        qubit_pairs: List of qubit-pair objects with a ``.name`` attribute.
        exchange_duration: Fixed exchange pulse duration (ns) — used when
            *duration_array* is ``None``.
        duration_array: Per-amplitude durations (ns) from the T_2π model.
            When provided, ``optimal_duration`` is looked up from this array.
        analysis_signal: Variable prefix to read from ``ds_raw``.

    Returns:
        ``(ds_fit, fit_results)`` where ``ds_fit`` contains
        ``mean_abs_diff_{name}`` (the amplitude-indexed mean |D|, minimised at V*)
        and ``diff_2d_{name}`` (the full 2-D difference), and ``fit_results``
        maps pair name → result dict.
    """
    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    phases = ds_raw.coords["analysis_phase"].values.astype(np.float64)

    fit_results: dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s; skipping.", var_name, qp.name)
            result = AmplitudePhaseFitResult(exchange_duration=float(exchange_duration))
            fit_results[qp.name] = asdict(result)
            continue

        data = ds_raw[var_name]

        if "analysis_phase" not in data.dims:
            logger.warning(
                "%s has no 'analysis_phase' dimension for pair %s; skipping.",
                var_name,
                qp.name,
            )
            result = AmplitudePhaseFitResult(exchange_duration=float(exchange_duration))
            fit_results[qp.name] = asdict(result)
            continue

        s0 = data.sel(control_state=0).values.astype(np.float64)  # (amplitude, phase)
        s1 = data.sel(control_state=1).values.astype(np.float64)  # (amplitude, phase)

        diff = s1 - s0  # (amplitude, phase)

        # Mean absolute parity difference ∝ |cos(J·t/2)| — minimised at CZ.
        mean_abs_diff = np.mean(np.abs(diff), axis=-1)  # (amplitude,)

        optimal_idx = int(np.argmin(mean_abs_diff))
        optimal_amplitude = float(amplitudes[optimal_idx])
        min_mad = float(mean_abs_diff[optimal_idx])

        if duration_array is not None:
            opt_dur = int(duration_array[optimal_idx])
        else:
            opt_dur = int(exchange_duration)

        # Success: the minimum must be in the interior of the sweep (not at a boundary).
        success = bool(
            len(amplitudes) >= 3
            and np.isfinite(optimal_amplitude)
            and amplitudes[0] < optimal_amplitude < amplitudes[-1]
        )

        result = AmplitudePhaseFitResult(
            exchange_duration=float(exchange_duration),
            optimal_amplitude=optimal_amplitude,
            optimal_duration=opt_dur,
            min_mean_abs_diff=min_mad,
            success=success,
        )
        fit_results[qp.name] = asdict(result)

        fit_vars[f"mean_abs_diff_{qp.name}"] = xr.DataArray(
            mean_abs_diff,
            dims=["exchange_amplitude"],
            coords={"exchange_amplitude": amplitudes},
            attrs={
                "long_name": "mean absolute parity difference ⟨|S(ctrl |1⟩)−S(ctrl |0⟩)|⟩_θ",
                "units": "",
            },
        )
        fit_vars[f"diff_2d_{qp.name}"] = xr.DataArray(
            diff,
            dims=["exchange_amplitude", "analysis_phase"],
            coords={
                "exchange_amplitude": amplitudes,
                "analysis_phase": phases,
            },
            attrs={
                "long_name": "parity difference S(ctrl |1⟩)−S(ctrl |0⟩)",
                "units": "",
            },
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={
            "exchange_amplitude": amplitudes,
            "analysis_phase": phases,
        },
        attrs={
            "exchange_duration": float(exchange_duration),
            "per_amplitude_durations": duration_array is not None,
        },
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log amplitude-phase calibration results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        dur_str = f"{r['optimal_duration']} ns" if r.get("optimal_duration") else f"{r['exchange_duration']:.0f} ns"
        _log(
            f"  {name}: [{status}] "
            f"V* = {r['optimal_amplitude']:.4f} V, "
            f"t(V*) = {dur_str}, "
            f"⟨|D|⟩(V*) = {r['min_mean_abs_diff']:.4f}"
        )