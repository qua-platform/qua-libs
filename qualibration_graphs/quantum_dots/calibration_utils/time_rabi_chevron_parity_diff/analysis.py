"""Chevron analysis via per-slice FFT peak detection.

Extracts f_res, t_π (= π/Ω), and γ (decay rate ≈ 1/T₂*) from per-frequency-
slice FFTs without assuming a specific pulse envelope.  Works for both square
and Gaussian pulses.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, Any

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode

from calibration_utils.time_rabi_chevron_parity_diff.init_utils import (
    _estimate_f_res_and_omega_from_chevron,
)

_logger = logging.getLogger(__name__)


@dataclass
class FitParameters:
    """Fit parameters for a single qubit from the Rabi chevron."""

    optimal_frequency: float  # Hz
    optimal_duration: float  # ns (π-time)
    rabi_frequency: float  # rad/ns
    decay_rate: float  # 1/ns  (γ ≈ 1/T₂*)
    success: bool


def _is_absolute_frequency(detuning_coord: np.ndarray) -> bool:
    """Heuristic: values > 0.5 GHz suggest absolute frequency."""
    return np.abs(detuning_coord).max() > 0.5e9


def _get_drive_frequencies_hz(
    ds: xr.Dataset,
    qubit: Any,
) -> np.ndarray:
    """Drive frequencies in Hz (from detuning coord; handles relative/absolute)."""
    detuning = np.asarray(ds.detuning.values, dtype=float)
    if _is_absolute_frequency(detuning):
        return detuning
    nominal = getattr(qubit.xy, "intermediate_frequency", 0.0)
    return nominal + detuning


def _fft_analyse_single_qubit(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Analyse one qubit's chevron via per-slice FFT.

    Returns (result_dict, fit_surface_2d) where fit_surface_2d is NaN
    (no 2D model reconstruction).
    """
    f_min, f_max = float(freqs_hz.min()), float(freqs_hz.max())

    try:
        f_res, omega, gamma = _estimate_f_res_and_omega_from_chevron(pdiff, freqs_hz, durations_ns, nominal_freq_hz)
    except Exception as exc:
        _logger.warning("FFT analysis failed: %s", exc)
        return {
            "optimal_frequency": nominal_freq_hz,
            "optimal_duration": np.nan,
            "rabi_frequency": np.nan,
            "decay_rate": np.nan,
            "success": False,
        }, np.full_like(pdiff, np.nan)

    t_pi = np.pi / omega if omega > 1e-12 else np.nan
    success = f_min <= f_res <= f_max and np.isfinite(t_pi) and np.isfinite(f_res)

    return {
        "optimal_frequency": float(f_res),
        "optimal_duration": float(t_pi),
        "rabi_frequency": float(omega),
        "decay_rate": float(gamma),
        "success": success,
    }, np.full_like(pdiff, np.nan)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add full_freq coord (nominal + detuning) from first qubit for plotting."""
    qubits = node.namespace["qubits"]
    if qubits:
        f = _get_drive_frequencies_hz(ds, qubits[0])
        ds = ds.assign_coords(full_freq=(["detuning"], f))
        ds.full_freq.attrs = {"long_name": "drive frequency", "units": "Hz"}
    return ds


def _get_qubit_names_for_fit(ds: xr.Dataset, qubits: list) -> list[str]:
    """Resolve qubit names that match ds data vars (pdiff_{name})."""
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    if not pdiff_vars:
        return []
    # Use names from ds (pdiff_Q1 -> Q1) so fit_results keys match the dataset
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit f_res and t_π per qubit. Returns (ds_fit, fit_results)."""
    qubits = node.namespace["qubits"]
    qubit_names = _get_qubit_names_for_fit(ds, qubits)
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]
    qubits_by_name = {getattr(q, "name", f"Q{i}"): q for i, q in enumerate(qubits)}
    if not qubits_by_name and qubit_names:
        qubits_by_name = dict(zip(qubit_names, list(qubits)[: len(qubit_names)]))
    durations_ns = np.asarray(ds.pulse_duration.values, dtype=float)

    fit_results = {}
    fit_arrays = {}

    for qname in qubit_names:
        qubit = qubits_by_name.get(qname)
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            nominal = (
                float(np.asarray(ds.detuning).mean())
                if qubit is None
                else getattr(qubit.xy, "intermediate_frequency", 0.0)
            )
            fp = FitParameters(
                optimal_frequency=nominal,
                optimal_duration=np.nan,
                rabi_frequency=np.nan,
                decay_rate=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        freqs_hz = _get_drive_frequencies_hz(ds, qubit) if qubit else np.asarray(ds.detuning.values, dtype=float)
        nominal_freq = (
            getattr(qubit.xy, "intermediate_frequency", float(freqs_hz.mean())) if qubit else float(freqs_hz.mean())
        )

        result, fit_surface = _fft_analyse_single_qubit(pdiff, freqs_hz, durations_ns, nominal_freq)

        fp = FitParameters(
            optimal_frequency=result["optimal_frequency"],
            optimal_duration=result["optimal_duration"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result.get("decay_rate", np.nan),
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_arrays[f"pdiff_{qname}_fit"] = (["detuning", "pulse_duration"], fit_surface)

    ds_fit = ds.assign(**fit_arrays)
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        f_res = r.get("optimal_frequency", 0) * 1e-9
        t_pi = r.get("optimal_duration", 0)
        gamma = r.get("decay_rate", 0)
        t2_star = 1.0 / gamma if gamma > 0 else float("inf")
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"f_res={f_res:.4f} GHz, "
            f"t_π={t_pi:.0f} ns, "
            f"γ={gamma:.5f} 1/ns (T₂*={t2_star:.0f} ns), "
            f"success={success}"
        )
        log_callable(msg)
