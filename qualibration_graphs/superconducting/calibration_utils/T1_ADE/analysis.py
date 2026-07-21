import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

logger = logging.getLogger(__name__)

# QUA `fixed` ~ [-8, 8). Shared with the QUA program in T1_ADE.py.
QUA_FIXED_MAX = 8.0
SAFE_CEILING = QUA_FIXED_MAX - 2.0
TIME_SCALE_US = 4.0 * QUA_FIXED_MAX
LN_ARG_FLOOR = float(np.exp(-QUA_FIXED_MAX))
SQRT_ARG_FLOOR = (LN_ARG_FLOOR + 0.5) ** 2
GAMMA_ADAPTIVE_FLOOR = 1.0 / TIME_SCALE_US
DENOM_SQ_FLOOR = 1.0 / SAFE_CEILING
GAMMA_FLOOR = 1e-4
DENOM_MIN = float(np.sqrt(DENOM_SQ_FLOOR))


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


def fetch_raw_dataset(job, qubits, n_reps: int, n_avg: int) -> xr.Dataset:
    """Fetch all ADE streams after job completion and merge into one dataset."""
    handles = job.result_handles
    repetition = np.arange(1, n_reps + 1)
    rep_axis = {"repetition": repetition}
    shot_axis = {"shot": np.arange(n_avg), "repetition": repetition}

    ds_parts = [
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "estimated_gamma"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "sigma_gamma"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "dt_used"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P0"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P1"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "P3"),
        fetch_results_as_xarray_arb_var(handles, qubits, rep_axis, "time_stamp"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "shots0_"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "shots1_"),
        fetch_results_as_xarray_arb_var(handles, qubits, shot_axis, "shots3_"),
    ]
    return xr.merge(ds_parts)


@dataclass
class T1ADEFit:
    """Stores the relevant T1 ADE analysis parameters for a single qubit."""

    sigma_T1_us: np.ndarray
    sigma_T1_boot_us: np.ndarray
    clipped: np.ndarray


def ade_gamma1_from_P(P0_, P1_, P3_, dt_us, sqrt_arg_floor=SQRT_ARG_FLOOR, ln_arg_floor=LN_ARG_FLOOR):
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


def ade_point_clipped(P0_, P1_, P3_, denom_min=DENOM_MIN, sqrt_arg_floor=SQRT_ARG_FLOOR, ln_arg_floor=LN_ARG_FLOOR):
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


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, T1ADEFit]]:
    """Compute T1, uncertainty bands, and clipping flags for each qubit."""
    qubits = node.namespace["qubits"]
    log_callable = getattr(node, "log", None)
    return _extract_relevant_fit_parameters(
        ds,
        qubits,
        node.parameters.num_repetitions,
        node.parameters.n_avg_per_point,
        node.parameters.n_bootstrap,
        log_callable=log_callable,
    )


def _analytical_sigma_T1(ds, qubits, gamma_floor=GAMMA_FLOOR):
    clipped_by_qubit = {}
    sigma_T1_by_qubit = {}
    for qubit in qubits:
        qubit_name = qubit.name
        P0_v = ds.P0.sel(qubit=qubit_name).values
        P1_v = ds.P1.sel(qubit=qubit_name).values
        P3_v = ds.P3.sel(qubit=qubit_name).values
        gamma_v = ds.estimated_gamma.sel(qubit=qubit_name).values
        sigma_gamma_v = ds.sigma_gamma.sel(qubit=qubit_name).values
        clipped_v = ade_point_clipped(P0_v, P1_v, P3_v)
        clipped_v |= ~np.isfinite(gamma_v) | (gamma_v <= gamma_floor)
        clipped_by_qubit[qubit_name] = clipped_v

        valid = np.isfinite(gamma_v) & (gamma_v > gamma_floor)
        sigma_T1_v = np.divide(
            sigma_gamma_v,
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
    gamma_floor=GAMMA_FLOOR,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    sigma_T1_boot_by_qubit = {}
    for qubit in qubits:
        qubit_name = qubit.name
        shots0_v = ds["shots0_"].sel(qubit=qubit_name).values
        shots1_v = ds["shots1_"].sel(qubit=qubit_name).values
        shots3_v = ds["shots3_"].sel(qubit=qubit_name).values
        dt_v = ds.dt_used.sel(qubit=qubit_name).values
        clipped_v = clipped_by_qubit[qubit_name]

        n_r = shots0_v.shape[0]
        sigma_T1_boot_v = np.full(n_r, np.nan)
        for r in range(n_r):
            idx0 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            idx1 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            idx3 = rng.integers(0, n_avg, size=(n_bootstrap, n_avg))
            P0_boot = shots0_v[r][idx0].mean(axis=1)
            P1_boot = shots1_v[r][idx1].mean(axis=1)
            P3_boot = shots3_v[r][idx3].mean(axis=1)
            gamma_boot = ade_gamma1_from_P(P0_boot, P1_boot, P3_boot, dt_v[r])
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


def _time_to_decision_stats(time_stamp, n_reps, n_avg_per_point, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

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
    log_callable=None,
) -> Tuple[xr.Dataset, dict[str, T1ADEFit]]:
    """Add metadata to the dataset and per-qubit fit results."""
    clipped_by_qubit, sigma_T1_by_qubit = _analytical_sigma_T1(ds, qubits)
    sigma_T1_boot_by_qubit = _bootstrap_sigma_T1(
        ds, qubits, n_avg, n_bootstrap, clipped_by_qubit,
    )

    ds["estimated_T1"] = 1.0 / ds.estimated_gamma
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

    time_to_decision_ms, mean_dt = _time_to_decision_stats(
        ds.time_stamp, n_reps, n_avg, log_callable=log_callable,
    )
    ds.attrs["time_to_decision_ms"] = time_to_decision_ms
    ds.attrs["mean_dt_s"] = mean_dt

    fit_results = {
        q.name: T1ADEFit(
            sigma_T1_us=ds.sigma_T1.sel(qubit=q.name).values,
            sigma_T1_boot_us=ds.sigma_T1_boot.sel(qubit=q.name).values,
            clipped=ds.clipped.sel(qubit=q.name).values,
        )
        for q in qubits
    }
    return ds, fit_results
