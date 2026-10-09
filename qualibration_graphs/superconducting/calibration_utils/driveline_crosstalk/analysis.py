"""Fit magnitude, calibrate destructive phase, and check independent data."""

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation, oscillation


def fit_amplitude(ds, parameters):
    names = list(ds.qubit.values)
    ds = ds.copy()
    scale = xr.DataArray(
        [[parameters.self_drive_amp_scale if t == d else 1 for d in names] for t in names],
        dims=("qubit", "drive_qubit"),
        coords={"qubit": names, "drive_qubit": names},
    )
    ds["pair_amp_scale"] = scale
    fits = np.full((len(names), len(names), 4), np.nan)
    quality = np.full((len(names), len(names)), np.nan)
    counts = np.zeros((len(names), len(names)), dtype=int)
    for j, target in enumerate(names):
        for i, drive in enumerate(names):
            trace = ds.state.sel(qubit=[target], drive_qubit=[drive]).reset_coords(drop=True)
            if not np.all(np.isfinite(trace)) or np.ptp(trace.values) < 0.1:
                continue
            fit = fit_oscillation(trace, "amp_prefactor")
            f = abs(float(fit.sel(fit_vals="f").item()))
            # Use the initial four periods when the full sweep contains many
            # periods: high-power nonlinearity must not bias the ratio seed.
            stop = min(float(ds.amp_prefactor.max()), 4 / f) if f > 0 else float(ds.amp_prefactor.max())
            short = trace.sel(amp_prefactor=slice(0, stop))
            if short.sizes["amp_prefactor"] >= 16:
                trace = short
                fit = fit_oscillation(trace, "amp_prefactor")
            values = [float(fit.sel(fit_vals=k).item()) for k in ("a", "f", "phi", "offset")]
            prediction = oscillation(trace.amp_prefactor.values, *values)
            r2 = 1 - np.sum((trace.values - prediction) ** 2) / np.sum((trace.values - trace.values.mean()) ** 2)
            quality[j, i], counts[j, i] = r2, trace.size
            # Frequency must be resolved by the sampled amplitudes.
            if r2 >= parameters.min_fit_r2 and 0 < abs(values[1]) < 0.5 / float(
                ds.amp_prefactor.diff("amp_prefactor").min()
            ):
                fits[j, i] = values
    result = ds[["target_rf_hz", "drive_base_amplitude", "drive_mV_per_prefactor"]].copy()
    result["fit"] = (("qubit", "drive_qubit", "fit_vals"), fits)
    result = result.assign_coords(fit_vals=["a", "f", "phi", "offset"])
    result["fit_r2"] = (("qubit", "drive_qubit"), quality)
    result["fit_points"] = (("qubit", "drive_qubit"), counts)
    result["pi_amplitude"] = ds.drive_base_amplitude * scale / (2 * abs(result.fit.sel(fit_vals="f")))
    diagonal = xr.DataArray(
        np.diag(result.pi_amplitude.transpose("qubit", "drive_qubit")), dims="qubit", coords={"qubit": names}
    )
    result["amplitude_ratio"] = (diagonal / result.pi_amplitude).transpose("drive_qubit", "qubit")
    result.amplitude_ratio.attrs["definition"] = (
        "a_pi(target->target) / a_pi(source->target); target pulse/source pulse"
    )
    result.attrs.update(ds.attrs)
    table = {
        f"{d}->{t}": {
            "amplitude_ratio": float(result.amplitude_ratio.sel(qubit=t, drive_qubit=d)),
            "fit_r2": float(result.fit_r2.sel(qubit=t, drive_qubit=d)),
        }
        for t in names
        for d in names
    }
    return ds, result, table


def select_phase(ds, magnitude, coarse=False):
    """Select a measured minimum; low amplitudes avoid coarse Rabi aliases."""
    names = list(ds.qubit.values)
    theta = xr.DataArray(
        np.full((len(names), len(names)), np.nan),
        dims=("drive_qubit", "qubit"),
        coords={"drive_qubit": names, "qubit": names},
    )
    for target in names:
        for drive in names:
            if target == drive:
                continue
            signal = ds.state.sel(qubit=target, drive_qubit=drive)
            if coarse:
                pi = float(magnitude.pi_amplitude.sel(qubit=target, drive_qubit=drive))
                base = float(ds.drive_base_amplitude.sel(drive_qubit=drive))
                # At most 1.5 source-only Rabi periods for the coarse phase.
                indices = np.flatnonzero(ds.amp_prefactor.values <= 3 * pi / base)
                signal = (
                    signal.isel(amp_prefactor=indices[: max(3, len(indices))])
                    if len(indices) >= 3
                    else signal.isel(amp_prefactor=slice(0, 3))
                )
            cost = signal.mean("amp_prefactor")
            index = int(cost.argmin("phase_index"))
            theta.loc[dict(qubit=target, drive_qubit=drive)] = (
                float(ds.phase_turns.sel(qubit=target, drive_qubit=drive).isel(phase_index=index)) % 1
            )
    return theta


def check_compensation(ds, magnitude, phase, parameters):
    result = magnitude.drop_vars([v for v in ("fit", "fit_points", "fit_r2", "pi_amplitude") if v in magnitude]).copy()
    result["compensation_phase_rad"] = phase.compensation_phase_rad
    result["compensation_phase_deg"] = np.rad2deg(phase.compensation_phase_rad)
    r, theta = result.amplitude_ratio, result.compensation_phase_rad
    diagonal = xr.DataArray(np.eye(r.sizes["drive_qubit"], dtype=bool), dims=r.dims, coords=r.coords)
    # K is the added target pulse relative to the source, not the leakage.
    result["compensation_real"] = (r * np.cos(theta)).where(~diagonal, 0)
    result["compensation_imag"] = (r * np.sin(theta)).where(~diagonal, 0)
    baseline = ds.state.sel(mode="no_drive", drop=True)
    result["no_drive_mean_population"] = baseline.mean("amp_prefactor")
    result["uncompensated_mean_population"] = ds.state.sel(mode="uncompensated", drop=True).mean("amp_prefactor")
    result["compensated_mean_population"] = ds.state.sel(mode="compensated", drop=True).mean("amp_prefactor")
    before = ds.state.sel(mode="uncompensated", drop=True) - baseline
    after = ds.state.sel(mode="compensated", drop=True) - baseline
    result["uncompensated_mean_excitation"] = before.mean("amp_prefactor")
    result["compensated_mean_residual"] = after.mean("amp_prefactor")
    result["compensated_max_abs_residual"] = abs(after).max("amp_prefactor")
    result["residual_fraction"] = abs(result.compensated_mean_residual) / result.uncompensated_mean_excitation
    noise = 3 * np.sqrt(0.5 / parameters.num_shots)
    result["validation_passed"] = (
        (
            (result.uncompensated_mean_excitation > noise)
            & (abs(result.compensated_mean_residual) <= parameters.max_mean_residual)
            & (result.residual_fraction <= parameters.max_residual_fraction)
            & (result.compensated_max_abs_residual <= noise)
        )
        .fillna(False)
        .astype(int)
    )
    result.attrs.update(
        ds.attrs,
        three_sigma_point_threshold=noise,
        coefficient_definition="K(source,target)=r*exp(i*compensation_phase_rad); additive compensation; diagonal zero",
        max_mean_residual=parameters.max_mean_residual,
        max_residual_fraction=parameters.max_residual_fraction,
    )
    table = {}
    for target in ds.qubit.values:
        for drive in ds.drive_qubit.values:
            if target == drive:
                continue
            selection = dict(qubit=target, drive_qubit=drive)
            table[f"{drive}->{target}"] = {
                name: float(result[name].sel(selection))
                for name in (
                    "amplitude_ratio",
                    "compensation_phase_rad",
                    "compensation_phase_deg",
                    "compensation_real",
                    "compensation_imag",
                    "no_drive_mean_population",
                    "uncompensated_mean_population",
                    "compensated_mean_population",
                    "uncompensated_mean_excitation",
                    "compensated_mean_residual",
                    "compensated_max_abs_residual",
                    "residual_fraction",
                )
            }
            table[f"{drive}->{target}"]["passed"] = bool(result.validation_passed.sel(selection))
    return result, table


def fit_phase(ds_raw: xr.Dataset, ds_fine: xr.Dataset):
    """Build the fitted compensation-phase matrix and directed-pair results."""
    magnitude = ds_raw
    theta = select_phase(ds_fine, magnitude)
    result = magnitude[["amplitude_ratio", "target_rf_hz", "drive_base_amplitude", "drive_mV_per_prefactor"]].copy()
    result["compensation_phase_rad"] = theta * 2 * np.pi
    result["compensation_phase_deg"] = theta * 360
    result["compensation_phase_turns"] = theta
    result.attrs.update(ds_raw.attrs, phase_convention="Destructive target phase; apply directly, no additional pi")
    fit_results = {
        f"{d}->{t}": {
            "amplitude_ratio": float(result.amplitude_ratio.sel(qubit=t, drive_qubit=d)),
            "compensation_phase_rad": float(result.compensation_phase_rad.sel(qubit=t, drive_qubit=d)),
            "compensation_phase_deg": float(result.compensation_phase_deg.sel(qubit=t, drive_qubit=d)),
        }
        for d in result.drive_qubit.values
        for t in result.qubit.values
        if d != t
    }
    return result, fit_results


def log_validation_results(fit_results, log_callable):
    """Log residual populations and cancellation status for every directed pair."""
    rows = fit_results
    log_callable("Pair             r     phase [deg]   mean residual   residual fraction   passed")
    for pair, row in rows.items():
        log_callable(
            f"{pair:10s} {row['amplitude_ratio']:.6f} {np.rad2deg(row['compensation_phase_rad']):10.3f} "
            f"{row['compensated_mean_residual']:14.5f} {row['residual_fraction']:18.3%}   {row['passed']}"
        )
