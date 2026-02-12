"""Chevron fit via damped Rabi with steady-state relaxation.

P(t,Δ) = A·(Ω/Ω_R)²·[sin²(Ω_R t/2)·e^{-γt} + ½(1-e^{-γt})] + offset.
Extracts f_res, t_π, γ (decay rate ≈ 1/T₂*).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, Any

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

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


def _rabi_chevron_model(
    params: np.ndarray,
    freq_hz: np.ndarray,
    duration_ns: np.ndarray,
) -> np.ndarray:
    """Damped Rabi with steady-state relaxation.

    P(t,Δ) = A·(Ω/Ω_R)²·[sin²(Ω_R t/2)·e^{-γt} + ½(1-e^{-γt})] + offset

    At early times (γt≪1) this reduces to the coherent Rabi formula.
    At late times the oscillation decays toward the detuning-dependent
    steady state A·(Ω/Ω_R)²·½ + offset, matching the physical behaviour
    of a driven dissipative two-level system.

    params = (f_res, omega, A, offset, gamma).
    Returns flat 1-D array of length ``len(freq_hz) * len(duration_ns)``.
    """
    f_res, omega, amplitude, offset, gamma = params
    f_2d, t_2d = np.meshgrid(freq_hz, duration_ns, indexing="ij")
    delta_2d = 2.0 * np.pi * (f_2d - f_res) * 1e-9  # rad/ns
    omega_R = np.sqrt(omega**2 + delta_2d**2)
    omega_R = np.where(omega_R > 1e-12, omega_R, 1e-12)
    visibility = (omega / omega_R) ** 2
    phase = omega_R * t_2d / 2.0
    decay = np.exp(-gamma * t_2d)
    # Oscillation decays toward steady-state ½ (time-averaged sin²)
    osc = np.sin(phase) ** 2 * decay + 0.5 * (1.0 - decay)
    return (amplitude * visibility * osc + offset).ravel()


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


def _fit_chevron_single_qubit(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit one qubit's chevron. Returns (result_dict, fit_surface_2d)."""
    y_flat = pdiff.ravel().astype(float)

    # Bounds and initial guess
    f_min, f_max = float(freqs_hz.min()), float(freqs_hz.max())
    omega_min = 2 * np.pi * 0.001  # rad/ns (~1 MHz Rabi)
    omega_max = 2 * np.pi * 0.5  # rad/ns (~500 MHz Rabi)
    f_res_init, omega_init, gamma_init = _estimate_f_res_and_omega_from_chevron(
        pdiff, freqs_hz, durations_ns, nominal_freq_hz
    )
    # Constrain γ around the heuristic estimate to prevent the optimizer
    # from trading amplitude for decay rate.  Allow a 10× window around
    # the init guess (floored at 1e-6, capped at 0.5).
    gamma_lo = max(gamma_init / 10.0, 1e-6) if gamma_init > 0 else 0.0
    gamma_hi = min(gamma_init * 10.0, 0.5) if gamma_init > 0 else 0.1
    _logger.debug("gamma_init=%.6f → bounds [%.6f, %.6f]", gamma_init, gamma_lo, gamma_hi)
    p0 = [
        f_res_init,
        omega_init,
        float(np.ptp(y_flat)) if np.ptp(y_flat) > 0 else 0.5,
        float(np.min(y_flat)),
        gamma_init,
    ]
    bounds = (
        [f_min - 1e6, omega_min, 0.0, -0.1, gamma_lo],
        [f_max + 1e6, omega_max, 2.0, 1.1, gamma_hi],
    )

    def _model_flat(_x: np.ndarray, f_res: float, om: float, A: float, off: float, gam: float) -> np.ndarray:
        """Wrapper for curve_fit; _x unused, uses freqs_hz/durations_ns from closure."""
        return _rabi_chevron_model(
            np.array([f_res, om, A, off, gam]),
            freqs_hz,
            durations_ns,
        )

    try:
        popt, _ = curve_fit(
            _model_flat,
            xdata=np.zeros(len(y_flat)),  # Unused; model uses freqs/durations from closure
            ydata=y_flat,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except Exception as exc:
        _logger.warning("curve_fit failed for chevron: %s", exc)
        return {
            "optimal_frequency": nominal_freq_hz,
            "optimal_duration": np.nan,
            "rabi_frequency": np.nan,
            "decay_rate": np.nan,
            "success": False,
        }, np.full_like(pdiff, np.nan)

    f_res, omega, amplitude, offset, gamma = popt
    t_pi_ns = np.pi / omega

    # Sanity checks
    success = f_min <= f_res <= f_max and 4 <= t_pi_ns <= 1e6 and np.isfinite(t_pi_ns) and np.isfinite(f_res)

    fit_surface = _rabi_chevron_model(popt, freqs_hz, durations_ns).reshape(pdiff.shape)
    return {
        "optimal_frequency": float(f_res),
        "optimal_duration": float(t_pi_ns),
        "rabi_frequency": float(omega),
        "decay_rate": float(gamma),
        "success": success,
    }, fit_surface


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

        use_numpyro = getattr(node.parameters, "use_numpyro", False)
        if use_numpyro:
            try:
                from calibration_utils.bayesian_utils import MCMCConfig
                from calibration_utils.time_rabi_chevron_parity_diff.analysis_numpyro import (
                    _fit_chevron_single_qubit_numpyro,
                )

                mcmc_config = MCMCConfig(
                    num_warmup=getattr(node.parameters, "mcmc_num_warmup", 500),
                    num_samples=getattr(node.parameters, "mcmc_num_samples", 500),
                    num_chains=getattr(node.parameters, "mcmc_num_chains", 1),
                )
                result, fit_surface = _fit_chevron_single_qubit_numpyro(
                    pdiff,
                    freqs_hz,
                    durations_ns,
                    nominal_freq,
                    config=mcmc_config,
                )
            except (ImportError, ModuleNotFoundError):
                _logger.warning("use_numpyro=True but bayesian extra not installed; falling back to scipy curve_fit")
                result, fit_surface = _fit_chevron_single_qubit(pdiff, freqs_hz, durations_ns, nominal_freq)
        else:
            result, fit_surface = _fit_chevron_single_qubit(pdiff, freqs_hz, durations_ns, nominal_freq)

        fp = FitParameters(
            optimal_frequency=result["optimal_frequency"],
            optimal_duration=result["optimal_duration"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result.get("decay_rate", np.nan),
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        # Propagate uncertainty estimates from NumPyro (when available)
        for extra_key in (
            "optimal_frequency_std",
            "optimal_duration_std",
            "decay_rate_std",
            "T2_star",
            "T2_star_std",
        ):
            if extra_key in result:
                fit_results[qname][extra_key] = result[extra_key]
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
