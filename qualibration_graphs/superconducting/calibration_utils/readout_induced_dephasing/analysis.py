"""Analysis and plotting for cross-Ramsey readout-induced dephasing."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    aggressor_qubit: str
    victim_qubit: str
    report_amp_factor: float
    readout_duration_ns: float
    extra_dephasing_rate_hz: float
    phase_flip_probability: float
    coherent_rotation_rad: float
    coherent_rotation_deg: float
    success: bool


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert raw IQ and attach physical aggressor readout coordinates."""
    if "state" not in ds:
        if not {"I", "Q"}.issubset(ds.data_vars):
            raise ValueError("Dataset must contain state or both I and Q")
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    by_name = {q.name: q for q in node.namespace["qubits"]}
    operation = node.parameters.operation
    amplitudes = [by_name[str(name)].resonator.operations[operation].amplitude for name in ds.aggressor_qubit.values]
    durations = [by_name[str(name)].resonator.operations[operation].length for name in ds.aggressor_qubit.values]
    ds = ds.assign_coords(
        readout_amplitude=(
            ("aggressor_qubit", "amp_factor"),
            np.asarray(amplitudes)[:, None] * ds.amp_factor.values[None, :],
        ),
        readout_duration_ns=("aggressor_qubit", durations),
    )
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}
    ds.readout_duration_ns.attrs = {"long_name": "readout pulse duration", "units": "ns"}
    return ds


def _fit_fringe(values: np.ndarray, frames: np.ndarray):
    finite = np.isfinite(values) & np.isfinite(frames)
    if finite.sum() < 4:
        return (np.nan,) * 5
    theta = 2 * np.pi * frames[finite]
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    offset, cosine, sine = np.linalg.lstsq(design, values[finite], rcond=None)[0]
    fitted = offset + cosine * np.cos(2 * np.pi * frames) + sine * np.sin(2 * np.pi * frames)
    amplitude = np.hypot(cosine, sine)
    contrast = 2 * amplitude
    phase = np.arctan2(-sine, cosine)
    return contrast, phase, offset, amplitude, fitted


def _gaussian_contrast(amp_factor, contrast_zero, dephasing_exponent):
    """Zero-centred Gaussian: C(a)=C0 exp[-lambda a^2]."""
    return contrast_zero * np.exp(-dephasing_exponent * np.asarray(amp_factor) ** 2)


def _fit_pair(signal: xr.DataArray, amp_factors: np.ndarray, frames: np.ndarray, duration_ns: float):
    contrasts, phases, offsets, fringe_amplitudes, fringe_fits = [], [], [], [], []
    for index in range(len(amp_factors)):
        result = _fit_fringe(np.asarray(signal.isel(amp_factor=index)), frames)
        contrast, phase, offset, fringe_amplitude, fitted = result
        contrasts.append(contrast)
        phases.append(phase)
        offsets.append(offset)
        fringe_amplitudes.append(fringe_amplitude)
        fringe_fits.append(fitted)

    contrasts = np.asarray(contrasts)
    phases = np.asarray(phases)
    finite = np.isfinite(contrasts) & (contrasts > 0)
    success = finite.sum() >= 3 and np.isfinite(duration_ns) and duration_ns > 0
    contrast_zero = dephasing_exponent = phase_intercept = phase_quadratic = np.nan
    if success:
        try:
            contrast_zero_guess = float(np.nanmax(contrasts[finite]))
            popt, _ = curve_fit(
                _gaussian_contrast,
                amp_factors[finite],
                contrasts[finite],
                p0=(contrast_zero_guess, 0.01),
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=10000,
            )
            contrast_zero, dephasing_exponent = map(float, popt)
            phase_finite = np.isfinite(phases)
            unwrapped = np.unwrap(phases[phase_finite])
            phase_quadratic, phase_intercept = np.polyfit(amp_factors[phase_finite] ** 2, unwrapped, 1)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            success = False

    duration_s = duration_ns * 1e-9
    gamma = dephasing_exponent * amp_factors**2 / duration_s
    probability = (1 - np.exp(-gamma * duration_s)) / 2
    phase_shift = phase_quadratic * amp_factors**2
    return {
        "contrast": contrasts,
        "phase": phases,
        "offset": np.asarray(offsets),
        "fringe_amplitude": np.asarray(fringe_amplitudes),
        "fringe_fit": np.asarray(fringe_fits),
        "contrast_fit": _gaussian_contrast(amp_factors, contrast_zero, dephasing_exponent),
        "phase_fit": phase_intercept + phase_quadratic * amp_factors**2,
        "extra_dephasing_rate_hz": gamma,
        "phase_flip_probability": probability,
        "coherent_rotation_rad": phase_shift,
        "dephasing_exponent": dephasing_exponent,
        "phase_quadratic": phase_quadratic,
        "success": success,
    }


def fit_raw_data(ds: xr.Dataset, node):
    """Fit every directed aggressor->victim pair and derive Gamma and P_phi."""
    signal_key = "state" if "state" in ds else "I"
    amp_factors = np.asarray(ds.amp_factor.values, dtype=float)
    frames = np.asarray(ds.frame.values, dtype=float)
    pair_results = {}
    for aggressor in ds.aggressor_qubit.values:
        duration_ns = float(ds.readout_duration_ns.sel(aggressor_qubit=aggressor))
        for victim in ds.qubit.values:
            key = (str(aggressor), str(victim))
            if aggressor == victim:
                continue
            pair_results[key] = _fit_pair(
                ds[signal_key].sel(aggressor_qubit=aggressor, qubit=victim),
                amp_factors,
                frames,
                duration_ns,
            )

    pair_dims = ("aggressor_qubit", "qubit")
    shape = (ds.sizes["aggressor_qubit"], ds.sizes["qubit"])
    coords = {"aggressor_qubit": ds.aggressor_qubit, "qubit": ds.qubit}

    def pair_array(field, extra_dims=(), extra_coords=None):
        extra_shape = tuple(ds.sizes[dim] for dim in extra_dims)
        values = np.full(shape + extra_shape, np.nan)
        for ai, aggressor in enumerate(ds.aggressor_qubit.values):
            for vi, victim in enumerate(ds.qubit.values):
                result = pair_results.get((str(aggressor), str(victim)))
                if result is not None:
                    values[(ai, vi)] = result[field]
        all_coords = dict(coords)
        if extra_coords:
            all_coords.update(extra_coords)
        return xr.DataArray(values, dims=pair_dims + extra_dims, coords=all_coords)

    ds_fit = ds.assign(
        fringe_contrast=pair_array("contrast", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        fringe_phase_rad=pair_array("phase", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        fringe_fit=pair_array(
            "fringe_fit",
            ("amp_factor", "frame"),
            {"amp_factor": ds.amp_factor, "frame": ds.frame},
        ),
        gaussian_contrast_fit=pair_array("contrast_fit", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        quadratic_phase_fit_rad=pair_array("phase_fit", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        extra_dephasing_rate_hz=pair_array("extra_dephasing_rate_hz", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        phase_flip_probability=pair_array("phase_flip_probability", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        coherent_rotation_rad=pair_array("coherent_rotation_rad", ("amp_factor",), {"amp_factor": ds.amp_factor}),
        dephasing_exponent=pair_array("dephasing_exponent"),
        phase_quadratic_rad=pair_array("phase_quadratic"),
    )
    ds_fit["coherent_rotation_deg"] = np.rad2deg(ds_fit.coherent_rotation_rad)
    ds_fit.attrs["report_amp_factor"] = float(node.parameters.report_amp_factor)
    ds_fit.phase_flip_probability.attrs = {"long_name": "phase-flip probability", "units": ""}
    ds_fit.extra_dephasing_rate_hz.attrs = {"long_name": "extra dephasing rate", "units": "Hz"}
    ds_fit.coherent_rotation_deg.attrs = {"long_name": "coherent Z rotation", "units": "deg"}

    report_amp = float(node.parameters.report_amp_factor)
    fit_results = {}
    for (aggressor, victim), result in pair_results.items():
        duration_ns = float(ds.readout_duration_ns.sel(aggressor_qubit=aggressor))
        exponent = result["dephasing_exponent"] * report_amp**2
        gamma = exponent / (duration_ns * 1e-9)
        rotation = result["phase_quadratic"] * report_amp**2
        fit_results[f"{aggressor}->{victim}"] = asdict(
            FitResult(
                aggressor_qubit=aggressor,
                victim_qubit=victim,
                report_amp_factor=report_amp,
                readout_duration_ns=duration_ns,
                extra_dephasing_rate_hz=float(gamma),
                phase_flip_probability=float((1 - np.exp(-exponent)) / 2),
                coherent_rotation_rad=float(rotation),
                coherent_rotation_deg=float(np.rad2deg(rotation)),
                success=bool(result["success"]),
            )
        )
    return ds_fit, fit_results


def log_fitted_results(fit_results, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for pair, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        log_callable(
            f"{pair}: {status}; at amplitude factor {result['report_amp_factor']:.3g}, "
            f"P_phi={100 * result['phase_flip_probability']:.4g}%, "
            f"Gamma={result['extra_dephasing_rate_hz']:.4g} Hz, "
            f"Delta_phi={result['coherent_rotation_deg']:.4g} deg"
        )


def summarize_fit_results(fit_results):
    """Aggregate successful off-diagonal pairs for comparison with device-level benchmarks."""
    successful = {pair: result for pair, result in fit_results.items() if result["success"]}
    if not successful:
        return {
            "num_successful_pairs": 0,
            "mean_phase_flip_probability": np.nan,
            "mean_abs_coherent_rotation_deg": np.nan,
            "max_phase_flip_probability": np.nan,
            "worst_pair": None,
        }
    worst_pair = max(successful, key=lambda pair: successful[pair]["phase_flip_probability"])
    return {
        "num_successful_pairs": len(successful),
        "mean_phase_flip_probability": float(
            np.mean([result["phase_flip_probability"] for result in successful.values()])
        ),
        "mean_abs_coherent_rotation_deg": float(
            np.mean([abs(result["coherent_rotation_deg"]) for result in successful.values()])
        ),
        "max_phase_flip_probability": float(successful[worst_pair]["phase_flip_probability"]),
        "worst_pair": worst_pair,
    }
