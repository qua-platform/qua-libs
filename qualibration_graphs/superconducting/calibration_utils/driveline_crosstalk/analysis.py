import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation

MODES = ["no_drive", "uncompensated", "compensated"]


@dataclass
class DrivelineCrosstalkFit:
    """Stores the drive-line crosstalk compensation of a single directed (drive -> target) pair"""

    drive_qubit: str
    target_qubit: str
    amplitude_ratio: float
    phase: float
    uncompensated_excitation: float
    mean_residual: float
    residual_fraction: float
    success: bool


def amplitude_factors(parameters) -> np.ndarray:
    """Return the normalized drive-amplitude sweep, from 0 to 1 (the maximum amplitude of each pair)."""
    return np.linspace(0, 1, parameters.amplitude_points)


def get_pair_context(qubits: List, parameters) -> Tuple[xr.Dataset, Dict[str, str]]:
    """
    Gather the per-pair quantities needed by the sequences and the analysis.

    Returns a dataset with dimensions ``qubit`` (target) and ``drive_qubit`` and a dictionary with the reason why
    each skipped directed pair cannot be measured. The diagonal (self-drive) is always measured.
    """
    names = [q.name for q in qubits]
    drive_amplitude, drive_scale, self_scale = [], [], []
    for q in qubits:
        pulse = q.xy.operations[parameters.operation]
        if "Square" not in type(pulse).__name__ or pulse.amplitude <= 0:
            raise ValueError(f"{q.name}: '{parameters.operation}' must be a square pulse with a positive amplitude.")
        # Amplitude scale of the operation that reaches max_drive_amplitude at the end of the normalized sweep
        scale = parameters.max_drive_amplitude / float(pulse.amplitude)
        if scale >= 2:
            raise ValueError(
                f"{q.name}: '{parameters.operation}' has an amplitude of {pulse.amplitude}, it cannot be scaled up to "
                f"max_drive_amplitude={parameters.max_drive_amplitude} (QUA amplitude scales are below 2). Use an "
                f"operation with a larger amplitude or reduce max_drive_amplitude."
            )
        drive_amplitude.append(float(pulse.amplitude))
        drive_scale.append(scale)
        if parameters.self_drive_amp_scale is None:
            # The rotation angle scales with the pulse area: the calibrated x180 gives the pi amplitude of the long
            # square probe. The self-drive sweep then spans ~1.5 Rabi periods (3 pi).
            pi_amplitude = abs(np.sum(q.xy.operations["x180"].calculate_waveform())) / parameters.pulse_length_ns
            self_scale.append(min(scale, 3 * pi_amplitude / float(pulse.amplitude)))
        else:
            self_scale.append(parameters.self_drive_amp_scale * scale)

    valid = np.eye(len(qubits), dtype=bool)
    # Amplitude scale of the operation on the drive line, multiplying the normalized sweep: rows are targets
    scale = np.tile(np.asarray(drive_scale), (len(qubits), 1))
    skipped = {}
    for j, target in enumerate(qubits):
        scale[j, j] = self_scale[j]
        for i, drive in enumerate(qubits):
            if drive is target:
                continue
            pair = f"{drive.name}->{target.name}"
            drive_if = target.xy.RF_frequency - drive.xy.LO_frequency
            if abs(drive_if) > parameters.max_drive_if_in_mhz * 1e6:
                skipped[pair] = f"IF of {drive_if / 1e6:.1f} MHz is outside +/-{parameters.max_drive_if_in_mhz} MHz"
            elif drive.xy.opx_output == target.xy.opx_output:
                skipped[pair] = "drive and target share the same output port"
            else:
                valid[j, i] = True

    coords = {"qubit": names, "drive_qubit": names}
    ds = xr.Dataset(coords=coords)
    ds["drive_base_amplitude"] = ("drive_qubit", drive_amplitude)
    ds["target_base_amplitude"] = ("qubit", drive_amplitude)
    # Stored as floats: integer frequencies in Hz overflow the int32 of the netCDF3 files of the node storage
    ds["target_rf_frequency"] = ("qubit", [float(q.xy.RF_frequency) for q in qubits])
    ds["drive_lo_frequency"] = ("drive_qubit", [float(q.xy.LO_frequency) for q in qubits])
    ds["pair_amp_scale"] = (("qubit", "drive_qubit"), scale)
    ds["pair_valid"] = (("qubit", "drive_qubit"), valid)
    return ds, skipped


def _fit_rabi(x, y, f_seed):
    """Power Rabi from the ground state: offset - a cos(2 pi f x), i.e. phase fixed at pi.

    A free phase lets the cosine slide and mis-place the pi amplitude. Returns (a, f, phi, offset) in the
    `oscillation` convention, NaN if the fit fails.
    """
    model = lambda x, a, f, offset: offset - a * np.cos(2 * np.pi * f * x)  # noqa: E731
    try:
        (a, f, offset), _ = curve_fit(
            model, x, y, p0=(0.5 * np.ptp(y), f_seed, y.mean()), bounds=([0, 0, -1], [1, np.inf, 2]), maxfev=20000
        )
    except (RuntimeError, ValueError):
        return [np.nan] * 4
    return [a, f, np.pi, offset]


def fit_amplitude(ds: xr.Dataset, min_fit_r2: float) -> xr.Dataset:
    """
    Fit every drive -> target power Rabi trace and derive the amplitude ratio.

    ``amplitude_ratio(target, drive) = a_pi(target -> target) / a_pi(drive -> target)``, with both pi amplitudes in
    output (DAC) units. It is the target-line amplitude, relative to the drive-line amplitude, that produces the same
    rotation of the target qubit, i.e. the magnitude of the compensation.
    """
    names = list(ds.qubit.values)
    max_f = 0.5 / float(ds.amp_prefactor.diff("amp_prefactor").min())
    fits = np.full((len(names), len(names), 4), np.nan)
    quality = np.full((len(names), len(names)), np.nan)
    for j, target in enumerate(names):
        for i, drive in enumerate(names):
            trace = ds.state.sel(qubit=[target], drive_qubit=[drive]).reset_coords(drop=True)
            if not bool(ds.pair_valid.sel(qubit=target, drive_qubit=drive)) or not np.all(np.isfinite(trace)):
                continue
            if np.ptp(trace.values) < 0.1:
                continue
            fit = fit_oscillation(trace, "amp_prefactor")
            f = abs(float(fit.sel(fit_vals="f").item()))
            # Fit only the first 1.5 periods: the Rabi period stretches at high power
            # and a full-range single-frequency fit underestimates the low-amplitude rate.
            stop = min(float(ds.amp_prefactor.max()), 1.5 / f) if f > 0 else float(ds.amp_prefactor.max())
            short = trace.sel(amp_prefactor=slice(0, stop))
            if short.sizes["amp_prefactor"] >= 16:
                trace = short
            y = trace.values.ravel()
            values = _fit_rabi(trace.amp_prefactor.values, y, f)
            prediction = oscillation(trace.amp_prefactor.values, *values)
            r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
            quality[j, i] = r2
            # The frequency must be resolved by the sampled amplitudes.
            if r2 >= min_fit_r2 and 0 < abs(values[1]) < max_f:
                fits[j, i] = values

    result = xr.Dataset(coords={"qubit": names, "drive_qubit": names, "fit_vals": ["a", "f", "phi", "offset"]})
    result["fit"] = (("qubit", "drive_qubit", "fit_vals"), fits)
    result["fit_r2"] = (("qubit", "drive_qubit"), quality)
    result["pi_amplitude"] = ds.drive_base_amplitude * ds.pair_amp_scale / (2 * abs(result.fit.sel(fit_vals="f")))
    self_pi = xr.DataArray(
        np.diag(result.pi_amplitude.transpose("qubit", "drive_qubit").values), dims="qubit", coords={"qubit": names}
    )
    result["amplitude_ratio"] = self_pi / result.pi_amplitude
    result.amplitude_ratio.attrs = {
        "long_name": "compensation amplitude ratio",
        "description": "a_pi(target->target) / a_pi(drive->target), in output (DAC) amplitude units",
    }
    return result


def coarse_phase_sweep(names: List[str], num_phases: int) -> xr.DataArray:
    """Full-period compensation phase sweep, in turns, identical for every pair."""
    return xr.DataArray(
        np.broadcast_to(np.arange(num_phases) / num_phases, (len(names), len(names), num_phases)).copy(),
        dims=("qubit", "drive_qubit", "phase_index"),
        coords={"qubit": names, "drive_qubit": names, "phase_index": np.arange(num_phases)},
    )


def fine_phase_sweep(center: xr.DataArray, num_phases: int) -> xr.DataArray:
    """Narrow compensation phase sweep, in turns, of +/- one coarse step around the coarse minimum of each pair."""
    offsets = xr.DataArray(np.linspace(-1, 1, num_phases) / num_phases, dims="phase_index")
    return (center.fillna(0) + offsets).assign_coords(phase_index=np.arange(num_phases))


def select_phase(ds: xr.Dataset, amplitude_fit: xr.Dataset, coarse: bool = False) -> xr.DataArray:
    """Select the compensation phase (in turns) that minimizes the target excitation of every pair.

    For the coarse sweep, only amplitudes up to 1.5 source-only Rabi periods are averaged to avoid aliasing.
    """
    names = list(ds.qubit.values)
    theta = xr.DataArray(
        np.full((len(names), len(names)), np.nan),
        dims=("qubit", "drive_qubit"),
        coords={"qubit": names, "drive_qubit": names},
    )
    for target in names:
        for drive in names:
            signal = ds.state.sel(qubit=target, drive_qubit=drive)
            if target == drive or not np.all(np.isfinite(signal)):
                continue
            if coarse:
                pi = float(amplitude_fit.pi_amplitude.sel(qubit=target, drive_qubit=drive))
                base = float(ds.drive_base_amplitude.sel(drive_qubit=drive))
                base *= float(ds.pair_amp_scale.sel(qubit=target, drive_qubit=drive))
                indices = np.flatnonzero(ds.amp_prefactor.values <= 3 * pi / base)
                signal = signal.isel(amp_prefactor=indices if len(indices) >= 3 else slice(0, 3))
            index = int(signal.mean("amp_prefactor").argmin("phase_index"))
            theta.loc[dict(qubit=target, drive_qubit=drive)] = (
                float(ds.phase_turns.sel(qubit=target, drive_qubit=drive).isel(phase_index=index)) % 1
            )
    return theta


def fit_raw_data(
    ds_amplitude: xr.Dataset, ds_phase: xr.Dataset, ds_check: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, DrivelineCrosstalkFit]]:
    """
    Fit the amplitude ratio and the compensation phase and validate the cancellation of every directed pair.

    The validation uses the amplitude ratio and phase stored in ``ds_check`` (the values that were actually played),
    so that re-analysing loaded data gives the same verdict.
    """
    ds_fit = fit_amplitude(ds_amplitude, node.parameters.min_fit_r2)
    ds_fit["fitted_phase"] = 2 * np.pi * select_phase(ds_phase.sel(phase_sweep="fine"), ds_fit)

    ds_fit["amplitude_ratio_used"] = ds_check.amplitude_ratio_used
    ds_fit["compensation_phase"] = ds_check.compensation_phase_used
    ds_fit.compensation_phase.attrs = {"long_name": "compensation phase", "units": "rad"}
    baseline = ds_check.state.sel(mode="no_drive")
    before = ds_check.state.sel(mode="uncompensated") - baseline
    after = ds_check.state.sel(mode="compensated") - baseline
    ds_fit["uncompensated_excitation"] = before.mean("amp_prefactor")
    ds_fit["mean_residual"] = after.mean("amp_prefactor")
    ds_fit["max_abs_residual"] = abs(after).max("amp_prefactor")
    ds_fit["residual_fraction"] = abs(ds_fit.mean_residual) / ds_fit.uncompensated_excitation
    noise = 3 * np.sqrt(0.5 / ds_check.attrs["num_shots"])
    ds_fit["success"] = (
        (ds_fit.uncompensated_excitation > noise)
        & (abs(ds_fit.mean_residual) <= node.parameters.max_mean_residual)
        & (ds_fit.residual_fraction <= node.parameters.max_residual_fraction)
        & (ds_fit.max_abs_residual <= noise)
    ).fillna(False)
    ds_fit.attrs["three_sigma_point_threshold"] = noise

    fit_results = {}
    for target in ds_fit.qubit.values:
        for drive in ds_fit.drive_qubit.values:
            if target == drive:
                continue
            fit = ds_fit.sel(qubit=target, drive_qubit=drive)
            fit_results[f"{drive}->{target}"] = DrivelineCrosstalkFit(
                drive_qubit=str(drive),
                target_qubit=str(target),
                amplitude_ratio=float(fit.amplitude_ratio_used),
                phase=float(fit.compensation_phase),
                uncompensated_excitation=float(fit.uncompensated_excitation),
                mean_residual=float(fit.mean_residual),
                residual_fraction=float(fit.residual_fraction),
                success=bool(fit.success),
            )
    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict[str, dict], log_callable=None):
    """
    Logs the amplitude ratio, compensation phase and cancellation verdict of every directed pair.

    Parameters:
    -----------
    fit_results : dict
        Fit results keyed by "drive->target", as produced by ``asdict`` of ``DrivelineCrosstalkFit``.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for pair, fit in fit_results.items():
        log_callable(
            f"Drive-line crosstalk {pair} : r = {fit['amplitude_ratio']:.5f}, "
            f"phase = {np.rad2deg(fit['phase']):.2f} deg, "
            f"residual = {fit['mean_residual']:.4f} ({100 * fit['residual_fraction']:.1f} % of "
            f"{fit['uncompensated_excitation']:.4f}) --> {'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
