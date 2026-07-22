import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

logger = logging.getLogger(__name__)


def _extract_stream_base(input_string: str) -> Optional[str]:
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)
    if index is not None:
        return input_string[:index]
    return None


def fetch_results_as_xarray_arb_var(handles, qubits, measurement_axis, var_name=None):
    """Fetch QOP result handles into an xarray Dataset (one variable rank at a time)."""
    if var_name is None:
        meas_vars = list(
            {
                _extract_stream_base(handle)
                for handle in handles.keys()
                if _extract_stream_base(handle) is not None
            }
        )
    else:
        meas_vars = [var_name]
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, qubit in enumerate(qubits)]
        for meas_var in meas_vars
    ]
    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)
    measurement_axis = dict(measurement_axis)
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}
    return xr.Dataset(
        {meas_var: ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )


def fetch_raw_dataset(
    job,
    qubits,
    n_reps: int,
    n_avg: int,
    t1_conventional_idle_ns: np.ndarray | None = None,
) -> xr.Dataset:
    """Fetch ADE streams and optional mid-run conventional T1 sweep."""
    handles = job.result_handles
    repetition = np.arange(1, n_reps + 1)
    rep_axis = {"repetition": repetition}
    shot_axis = {"shot": np.arange(n_avg), "repetition": repetition}

    ds_parts = [
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "gamma1"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "sigma_gamma1"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "dt_used"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P0"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P1"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P3"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "time_stamp"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "P0_shots"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "P1_shots"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "P3_shots"),
    ]
    ds = xr.merge(ds_parts)

    if t1_conventional_idle_ns is not None and handles.get("t1_state1") is not None:
        fetch_t0 = time.perf_counter()
        ds_t1 = fetch_results_as_xarray_arb_var(
            handles, qubits, {"idle_time": t1_conventional_idle_ns}, "t1_state"
        )
        ds_t1 = ds_t1.rename({"t1_state": "t1_conventional_state"})
        ds_t1.idle_time.attrs = {"long_name": "idle time", "units": "ns"}
        ds = xr.merge([ds, ds_t1])
        ds.attrs["fetch_ms"] = (time.perf_counter() - fetch_t0) * 1e3
        measurement_ms = conventional_measurement_ms_from_timestamps(handles, qubits)
        if measurement_ms is not None:
            ds.attrs["measurement_ms"] = measurement_ms

    return ds


def conventional_measurement_ms_from_timestamps(handles, qubits) -> float | None:
    """FPGA duration of the mid-run conventional T1 sweep from ``t1_conv_start/end`` streams."""
    if handles.get("t1_conv_start1") is None or handles.get("t1_conv_end1") is None:
        return None

    start_cycles = np.array(
        [np.squeeze(handles.get(f"t1_conv_start{i + 1}").fetch_all()) for i in range(len(qubits))],
        dtype=float,
    )
    end_cycles = np.array(
        [np.squeeze(handles.get(f"t1_conv_end{i + 1}").fetch_all()) for i in range(len(qubits))],
        dtype=float,
    )
    measurement_ms = float(np.nanmean(end_cycles - start_cycles) * 4e-6)
    if not np.isfinite(measurement_ms) or measurement_ms <= 0:
        return None
    return measurement_ms


def estimate_conventional_measurement_ms(ds: xr.Dataset, n_reps: int) -> float:
    """Fallback FPGA duration estimate from the inflated ADE lab-time gap at mid-run.

    The conventional sweep runs after the ADE timestamp at ``n == n_reps // 2``, so the
    gap to the next repetition equals ``T_conventional + T_ADE_rep``.
    """
    per_rep_dt = np.diff(ds.time_stamp.values, axis=1)
    mid_diff_idx = n_reps // 2
    if mid_diff_idx <= 0 or mid_diff_idx >= per_rep_dt.shape[1]:
        return np.nan

    inflated_dt = float(np.nanmean(per_rep_dt[:, mid_diff_idx]))
    mask = np.ones(per_rep_dt.shape[1], dtype=bool)
    mask[mid_diff_idx] = False
    mean_ade_dt = float(np.nanmean(per_rep_dt[:, mask]))
    return max(0.0, (inflated_dt - mean_ade_dt) * 1e3)


@dataclass
class T1ADEFit:
    """Stores ADE and optional mid-run conventional T1 fit parameters per qubit."""

    sigma_T1_us: np.ndarray
    sigma_T1_boot_us: np.ndarray
    clipped: np.ndarray
    t1_conventional_us: float | None = None
    t1_conventional_error_us: float | None = None
    t1_conventional_success: bool | None = None


def ade_gamma1_from_P(P0_, P1_, P3_, dt_us, *, sqrt_arg_floor, ln_arg_floor):
    """Host ADE gamma [1/us] with the same clip floors as the QUA program."""
    denom = P1_ - P0_
    valid_denom = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    c_ = np.full(np.shape(P0_), np.nan, dtype=float)
    with np.errstate(all="ignore"):
        np.divide(P3_ - P0_, denom, out=c_, where=valid_denom)
        sqrt_arg = np.maximum(c_ - 0.75, sqrt_arg_floor)
        x_ = np.maximum(np.sqrt(sqrt_arg) - 0.5, ln_arg_floor)
        gamma = -np.log(x_) / dt_us
    return np.where(valid_denom & np.isfinite(gamma) & (gamma > 0), gamma, np.nan)


def ade_point_clipped(P0_, P1_, P3_, *, denom_min, sqrt_arg_floor, ln_arg_floor):
    """Flag reps where FPGA P values hit an ADE clip floor (sigma there is unreliable)."""
    denom = P1_ - P0_
    bad = ~np.isfinite(denom) | (np.abs(denom) < denom_min)
    c_ = np.full(np.shape(P0_), np.nan, dtype=float)
    with np.errstate(all="ignore"):
        np.divide(P3_ - P0_, denom, out=c_, where=~bad)
    bad |= ~np.isfinite(c_) | (c_ - 0.75 < sqrt_arg_floor)
    sqrt_arg = np.maximum(c_ - 0.75, sqrt_arg_floor)
    x_ = np.sqrt(sqrt_arg) - 0.5
    bad |= x_ <= ln_arg_floor
    bad |= x_ >= 1.0 - ln_arg_floor
    return bad


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw ADE stream units and normalize laboratory time."""
    ds = ds.copy(deep=True)
    ds["dt_used"] = ds.dt_used * 4e-3
    ds.dt_used.attrs = {"long_name": "ADE wait dt", "units": "us"}

    timestamp_values = ds.time_stamp.values
    ds = ds.assign(time_stamp=(ds.time_stamp.dims, timestamp_values))
    time_stamp = (ds.time_stamp - ds.time_stamp.min(dim="repetition")) * 4e-9
    time_stamp.attrs = {"long_name": "Laboratory time", "units": "s"}
    ds = ds.drop_vars("time_stamp").assign(time_stamp=time_stamp)

    return ds


def _analytical_sigma_T1(ds, qubits, clip_floors):
    gamma_floor = clip_floors["gamma_floor"]
    clipped_by_qubit = {}
    sigma_T1_by_qubit = {}
    for qubit in qubits:
        qubit_name = qubit.name
        P0_v = ds.P0.sel(qubit=qubit_name).values
        P1_v = ds.P1.sel(qubit=qubit_name).values
        P3_v = ds.P3.sel(qubit=qubit_name).values
        gamma_v = ds.gamma1.sel(qubit=qubit_name).values
        sigma_gamma1_v = ds.sigma_gamma1.sel(qubit=qubit_name).values
        clipped_v = ade_point_clipped(
            P0_v,
            P1_v,
            P3_v,
            denom_min=clip_floors["denom_min"],
            sqrt_arg_floor=clip_floors["sqrt_arg_floor"],
            ln_arg_floor=clip_floors["ln_arg_floor"],
        )
        clipped_v |= ~np.isfinite(gamma_v) | (gamma_v <= gamma_floor)
        clipped_by_qubit[qubit_name] = clipped_v

        valid = np.isfinite(gamma_v) & (gamma_v > gamma_floor)
        sigma_T1_v = np.divide(
            sigma_gamma1_v,
            gamma_v**2,
            out=np.full_like(gamma_v, np.nan),
            where=valid,
        )
        sigma_T1_v[clipped_v] = np.nan
        sigma_T1_by_qubit[qubit_name] = sigma_T1_v

    return clipped_by_qubit, sigma_T1_by_qubit


def _bootstrap_sigma_T1(
    ds,
    qubits,
    n_avg,
    n_bootstrap,
    clipped_by_qubit,
    clip_floors,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    gamma_floor = clip_floors["gamma_floor"]
    sigma_T1_boot_by_qubit = {}
    for qubit in qubits:
        qubit_name = qubit.name
        P0_shots_v = ds["P0_shots"].sel(qubit=qubit_name).values
        P1_shots_v = ds["P1_shots"].sel(qubit=qubit_name).values
        P3_shots_v = ds["P3_shots"].sel(qubit=qubit_name).values
        dt_v = ds.dt_used.sel(qubit=qubit_name).values
        clipped_v = clipped_by_qubit[qubit_name]

        n_r = P0_shots_v.shape[0]
        sigma_T1_boot_v = np.full(n_r, np.nan)
        for r in range(n_r):
            idx0 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            idx1 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            idx3 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            P0_boot = P0_shots_v[r][idx0].mean(axis=1)
            P1_boot = P1_shots_v[r][idx1].mean(axis=1)
            P3_boot = P3_shots_v[r][idx3].mean(axis=1)
            gamma_boot = ade_gamma1_from_P(
                P0_boot, P1_boot, P3_boot, dt_v[r],
                sqrt_arg_floor=clip_floors["sqrt_arg_floor"],
                ln_arg_floor=clip_floors["ln_arg_floor"],
            )
            valid = np.isfinite(gamma_boot) & (gamma_boot > gamma_floor)
            T1_boot = np.divide(1.0, gamma_boot, out=np.full_like(gamma_boot, np.nan), where=valid)
            if valid.sum() < n_bootstrap // 10:
                continue
            sigma_T1_boot_v[r] = (
                np.percentile(T1_boot[valid], 84) - np.percentile(T1_boot[valid], 16)
            ) / 2
        sigma_T1_boot_v[clipped_v] = np.nan
        sigma_T1_boot_by_qubit[qubit_name] = sigma_T1_boot_v

    return sigma_T1_boot_by_qubit


def _time_to_decision_stats(
    time_stamp=None,
    n_reps=None,
    n_avg_per_point=None,
    *,
    conventional_ttd=None,
    log_callable=None,
):
    if log_callable is None:
        log_callable = logger.info

    if conventional_ttd is not None:
        measurement_ms = float(conventional_ttd.get("measurement_ms", np.nan))
        fetch_ms = float(conventional_ttd.get("fetch_ms", np.nan))
        analysis_ms = float(conventional_ttd.get("analysis_ms", np.nan))
        total_ms = float(np.nansum([measurement_ms, fetch_ms, analysis_ms]))
        stats = {
            "measurement_ms": measurement_ms,
            "fetch_ms": fetch_ms,
            "analysis_ms": analysis_ms,
            "total_ms": total_ms,
        }
        log_callable(
            "Conventional T1 time-to-decision: "
            f"measurement={measurement_ms:.2f} ms, "
            f"fetch={fetch_ms:.2f} ms, "
            f"analysis={analysis_ms:.2f} ms, "
            f"total={total_ms:.2f} ms"
        )
        return stats

    t_vals = time_stamp.values
    per_rep_dt = np.diff(t_vals, axis=1)
    dt = np.mean(per_rep_dt, axis=1)
    mean_dt = float(np.mean(dt))
    stats = {
        "mean": mean_dt * 1e3,
        "median": float(np.median(per_rep_dt) * 1e3),
        "std": float(np.std(per_rep_dt) * 1e3),
    }
    log_callable(
        f"Time-to-decision per repetition: mean={stats['mean']:.2f} ms, "
        f"median={stats['median']:.2f} ms, std={stats['std']:.2f} ms "
        f"(n_reps={n_reps}, n_avg_per_point={n_avg_per_point})"
    )
    return stats, mean_dt


def _extract_relevant_fit_parameters(
    ds: xr.Dataset,
    qubits,
    n_reps,
    n_avg,
    n_bootstrap,
    clip_floors,
    log_callable=None,
    conventional_fit_results: dict | None = None,
    conventional_analysis_ms: float | None = None,
) -> Tuple[xr.Dataset, dict[str, T1ADEFit], dict[str, float]]:
    """Extract ADE fit parameters and optional mid-run conventional T1 fit parameters."""
    clipped_by_qubit, sigma_T1_by_qubit = _analytical_sigma_T1(ds, qubits, clip_floors)
    sigma_T1_boot_by_qubit = _bootstrap_sigma_T1(
        ds, qubits, n_avg, n_bootstrap, clipped_by_qubit, clip_floors,
    )

    ds["estimated_T1"] = 1.0 / ds.gamma1
    ds.estimated_T1.attrs = {"long_name": "T1", "units": "us"}
    ds["sigma_T1"] = xr.DataArray(
        np.array([sigma_T1_by_qubit[q.name] for q in qubits]),
        coords={"qubit": [q.name for q in qubits], "repetition": ds.repetition},
    )
    ds.sigma_T1.attrs = {"long_name": "T1 uncertainty (analytical)", "units": "us"}
    ds["sigma_T1_boot"] = xr.DataArray(
        np.array([sigma_T1_boot_by_qubit[q.name] for q in qubits]),
        coords={"qubit": [q.name for q in qubits], "repetition": ds.repetition},
    )
    ds.sigma_T1_boot.attrs = {"long_name": "T1 uncertainty (bootstrap)", "units": "us"}
    ds["clipped"] = xr.DataArray(
        np.array([clipped_by_qubit[q.name] for q in qubits]),
        coords={"qubit": [q.name for q in qubits], "repetition": ds.repetition},
    )

    time_to_decision_ms, _ = _time_to_decision_stats(
        ds.time_stamp, n_reps, n_avg, log_callable=log_callable,
    )

    fit_results = {
        q.name: T1ADEFit(
            sigma_T1_us=ds.sigma_T1.sel(qubit=q.name).values,
            sigma_T1_boot_us=ds.sigma_T1_boot.sel(qubit=q.name).values,
            clipped=ds.clipped.sel(qubit=q.name).values,
        )
        for q in qubits
    }

    if conventional_fit_results is not None:
        ttd = {
            k: float(ds.attrs[k])
            for k in ("fetch_ms", "measurement_ms")
            if k in ds.attrs
        }
        if not np.isfinite(ttd.get("measurement_ms", np.nan)):
            ttd["measurement_ms"] = estimate_conventional_measurement_ms(ds, n_reps)
        ttd["analysis_ms"] = conventional_analysis_ms
        ttd = _time_to_decision_stats(conventional_ttd=ttd, log_callable=log_callable)
        for key, val in ttd.items():
            ds.attrs[key] = val

        for q in qubits:
            conv = conventional_fit_results[q.name]
            ade = fit_results[q.name]
            fit_results[q.name] = T1ADEFit(
                sigma_T1_us=ade.sigma_T1_us,
                sigma_T1_boot_us=ade.sigma_T1_boot_us,
                clipped=ade.clipped,
                t1_conventional_us=float(conv.t1) * 1e-3,
                t1_conventional_error_us=float(conv.t1_error) * 1e-3,
                t1_conventional_success=bool(conv.success),
            )

    return ds, fit_results, time_to_decision_ms


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, T1ADEFit], dict[str, float]]:
    """Fit mid-run conventional T1 (if present) and extract ADE + conventional parameters."""
    log_callable = getattr(node, "log", None)
    conventional_fit_results = None
    conventional_analysis_ms = None

    if "t1_conventional_state" in ds and node.parameters.measure_conventional_t1:
        from qualibration_libs.analysis import fit_decay_exp

        from calibration_utils.T1 import log_fitted_results
        from calibration_utils.T1.analysis import (
            _extract_relevant_fit_parameters as extract_t1_fit_parameters,
        )

        analysis_t0 = time.perf_counter()
        ds_t1 = ds[["t1_conventional_state"]].rename({"t1_conventional_state": "state"})
        fit_data = fit_decay_exp(ds_t1.state, "idle_time")
        ds_t1_fit = xr.merge([ds_t1, fit_data.rename("fit_data")])
        ds_t1_fit, conventional_fit_results = extract_t1_fit_parameters(ds_t1_fit)
        ds = xr.merge([ds, ds_t1_fit.drop_vars("state")])
        conventional_analysis_ms = (time.perf_counter() - analysis_t0) * 1e3
        log_fitted_results(ds_t1_fit, log_callable=log_callable)

    return _extract_relevant_fit_parameters(
        ds,
        node.namespace["qubits"],
        node.parameters.num_repetitions,
        node.parameters.n_avg_per_point,
        node.parameters.n_bootstrap,
        node.namespace["ade_clip_floors"],
        log_callable=log_callable,
        conventional_fit_results=conventional_fit_results,
        conventional_analysis_ms=conventional_analysis_ms,
    )
