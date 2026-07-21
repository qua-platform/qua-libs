from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from calibration_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qualibrate.core import QualibrationNode


def build_psb_readout_sweep(readout_length_min: int, readout_length_max: int, readout_length_points: int) -> dict:
    """Build sweep grid consistent with 05c charge-state readout time (arange + chunk step).

    QM requires the readout pulse length to equal an integer number of accumulated segments:
    ``pulse_length == num_segments * 4 * segment_length`` (see ``measure_accumulated``).
    ``readout_length_max`` is therefore rounded **down** to the nearest valid pulse length.

    Returns keys: ``array_size``, ``step_ns``, ``samples_per_chunk``, ``sweep_coord``,
    ``pulse_length`` (effective ns), ``segment_length`` (QUA ``segment_length`` arg),
    ``num_segments``.
    """
    r_min = max(4, int(readout_length_min) // 4 * 4)
    r_max = max(r_min + 4, int(readout_length_max) // 4 * 4)
    n_pts = max(1, int(readout_length_points))
    if n_pts < 2:
        step_ns = max(4, r_max - r_min)
    else:
        step_ns = max(4, ((r_max - r_min) // (n_pts - 1)) // 4 * 4)
    segment_length = max(1, step_ns // 4)
    chunk_ns = 4 * segment_length
    num_segments = r_max // chunk_ns
    if num_segments < 1:
        num_segments = 1
    pulse_length = num_segments * chunk_ns
    if pulse_length < r_min:
        num_segments = int(np.ceil(r_min / chunk_ns))
        pulse_length = num_segments * chunk_ns

    integrations_times = np.arange(r_min, pulse_length, step_ns, dtype=int)
    if len(integrations_times) == 0:
        integrations_times = np.array([min(r_min, pulse_length)], dtype=int)
    array_size = len(integrations_times)
    if array_size > num_segments:
        array_size = num_segments
        integrations_times = integrations_times[:array_size]
    sweep_coord = np.arange(1, array_size + 1, dtype=int) * chunk_ns
    return {
        "array_size": array_size,
        "step_ns": step_ns,
        "samples_per_chunk": segment_length,
        "sweep_coord": sweep_coord,
        "pulse_length": pulse_length,
        "segment_length": segment_length,
        "num_segments": num_segments,
    }


def _resolve_qubit_pairs(node: "QualibrationNode") -> List:
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def _duration_unit(t_ns: float, t_min: float, t_max: float) -> float:
    span = t_max - t_min
    if abs(span) < 1e-15:
        return 0.5
    return float(np.clip((t_ns - t_min) / span, 0.0, 1.0))


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _global_readout_axis_from_endpoints(I: np.ndarray, Q: np.ndarray, *, n_edge: int) -> tuple[float, float]:
    n_s = I.shape[1]
    k = max(1, min(n_edge, n_s // 2 or 1))
    I_lo = I[:, :k].ravel()
    Q_lo = Q[:, :k].ravel()
    I_hi = I[:, -k:].ravel()
    Q_hi = Q[:, -k:].ravel()
    delta = np.array(
        [np.nanmean(I_lo) - np.nanmean(I_hi), np.nanmean(Q_lo) - np.nanmean(Q_hi)],
        dtype=float,
    )
    norm = float(np.hypot(delta[0], delta[1]))
    if not np.isfinite(norm) or norm < 1e-20:
        return 1.0, 0.0
    delta /= norm
    return float(delta[0]), float(delta[1])


def plot_simulated_dataset_histograms(
    ds: xr.Dataset,
    *,
    sweep_name: str = "readout_length",
    qubit_pairs: Optional[Sequence[Union[str, object]]] = None,
    n_bins: int = 48,
    log_counts: bool = True,
    edge_fraction: float = 1.0 / 3.0,
) -> "Figure":
    """2D shot-density map vs readout-length sweep (raw, no Barthel fit)."""
    import matplotlib.pyplot as plt

    if sweep_name not in ds.dims and sweep_name not in ds.coords:
        raise KeyError(f"sweep_name={sweep_name!r} not found on ds")

    if qubit_pairs is None:
        names = [str(x) for x in ds["qubit_pair"].values]
    else:
        names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]

    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    sweep_vals = np.asarray(ds[sweep_name].values, dtype=float)
    attrs = ds[sweep_name].attrs if sweep_name in ds.coords else {}
    long_name = attrs.get("long_name", sweep_name)
    units = attrs.get("units")
    x_label = f"{long_name} [{units}]" if units else long_name

    n_s = len(sweep_vals)
    n_edge = max(1, int(np.ceil(n_s * edge_fraction)))

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        I = np.asarray(ds.I.sel(qubit_pair=name).values, dtype=float)
        Q = np.asarray(ds.Q.sel(qubit_pair=name).values, dtype=float)
        if I.ndim != 2 or Q.shape != I.shape:
            raise ValueError(f"Expected I,Q 2D (n_runs, {sweep_name}); got {I.shape}")

        c, s = _global_readout_axis_from_endpoints(I, Q, n_edge=n_edge)
        proj_cols = [I[:, si] * c + Q[:, si] * s for si in range(n_s)]
        flat = np.concatenate(proj_cols) if proj_cols else np.array([0.0])
        lo, hi = np.nanpercentile(flat, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        y_min, y_max = lo - pad, hi + pad
        edges = np.linspace(y_min, y_max, n_bins + 1)

        H = np.zeros((n_s, n_bins), dtype=float)
        for si in range(n_s):
            H[si, :], _ = np.histogram(proj_cols[si], bins=edges, density=False)

        Z = np.log10(H.T + 1.0) if log_counts else H.T

        if n_s == 1:
            dv = 1e-6 * (abs(float(sweep_vals[0])) + 1.0)
            x_left = float(sweep_vals[0]) - 0.5 * dv
            x_right = float(sweep_vals[0]) + 0.5 * dv
        else:
            x_left = float(sweep_vals[0])
            x_right = float(sweep_vals[-1])

        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=[x_left, x_right, y_min, y_max],
            interpolation="nearest",
            cmap="magma",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Projected IQ (endpoint centroids)")
        ax.set_title(name)
        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="log10(count+1)" if log_counts else "counts",
        )

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Shot histograms vs {sweep_name} (raw, no Barthel fit)")
    fig.tight_layout()
    return fig


def generate_simulated_dataset(node: "QualibrationNode") -> xr.Dataset:
    """Synthetic PSB readout-length sweep; shot noise scales down ~ 1/sqrt(T/T_ref)."""
    qubit_pairs = _resolve_qubit_pairs(node)
    pair_names = [qp.name for qp in qubit_pairs]

    readout_max = node.parameters.readout_length_max
    if readout_max is None:
        readout_max = (
            qubit_pairs[0].quantum_dot_pair.sensor_dots[0].readout_resonator.operations[
                "readout"
            ].length
        )

    sweep = build_psb_readout_sweep(
        node.parameters.readout_length_min,
        readout_max,
        node.parameters.readout_length_points,
    )
    readout_times_ns = sweep["sweep_coord"].astype(float)
    t_min = float(readout_times_ns.min()) if len(readout_times_ns) else 1.0
    t_max = float(readout_times_ns.max()) if len(readout_times_ns) else 1.0
    sweep_name = node.parameters.sweep_name

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in qubit_pairs]
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(pair_names),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        sweep_name: xr.DataArray(readout_times_ns, attrs={"long_name": "readout length", "units": "ns"}),
    }

    num_shots = int(node.parameters.num_shots)
    n_pairs = len(qubit_pairs)
    n_t = len(readout_times_ns)

    I_arr = np.zeros((n_pairs, num_shots, n_t), dtype=float)
    Q_arr = np.zeros((n_pairs, num_shots, n_t), dtype=float)

    tau_M = 1.0
    T1 = 2.0
    sigma_I_base = 0.12e-2
    sigma_Q_base = 0.10e-2
    t_ref = max(t_max, 1.0)

    for pi, _qp in enumerate(qubit_pairs):
        rng = np.random.default_rng(seed=42 + pi * 9973)
        theta = 0.38 + 0.27 * float(pi)
        c_ax, s_ax = float(np.cos(theta)), float(np.sin(theta))
        # Fixed PSB-open point (mid "duration" axis): separation along readout axis
        y_s = (-1.2e-2) * (1.0 + 0.06 * float(pi))
        y_t = (1.2e-2) * (1.0 + 0.06 * float(pi))
        mu_s = (y_s * c_ax, y_s * s_ax)
        mu_t = (y_t * c_ax, y_t * s_ax)

        for ti, t_ns in enumerate(readout_times_ns):
            u = _duration_unit(float(t_ns), t_min, t_max)
            scale = float(np.sqrt(max(t_ns / t_ref, 1e-6)))
            sep = 0.15 + 0.85 * u
            mu_s_t = (mu_s[0] * sep, mu_s[1] * sep)
            mu_t_t = (mu_t[0] * sep, mu_t[1] * sep)
            sigma_I = sigma_I_base / scale
            sigma_Q = sigma_Q_base / scale
            params = SimulationParamsIQ(
                n_samples=num_shots,
                p_triplet=0.5,
                mu_S=mu_s_t,
                mu_T=mu_t_t,
                sigma_I=sigma_I,
                sigma_Q=sigma_Q,
                rho=0.0,
                tau_M=tau_M,
                T1=T1,
            )
            X, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
            I_arr[pi, :, ti] = X[:, 0]
            Q_arr[pi, :, ti] = X[:, 1]

    dims = ("qubit_pair", "n_runs", sweep_name)
    return xr.Dataset(
        {"I": (dims, I_arr), "Q": (dims, Q_arr)},
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            sweep_name: xr.DataArray(
                readout_times_ns, dims=sweep_name, attrs={"long_name": "readout length", "units": "ns"}
            ),
        },
    )
