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


def _resolve_qubit_pairs(node: "QualibrationNode") -> List:
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def _detuning_unit(d: float, d_min: float, d_max: float) -> float:
    """Map detuning to ``t in [0,1]`` along the sweep (0 = ``d_min``, 1 = ``d_max``)."""
    span = d_max - d_min
    if abs(span) < 1e-15:
        return 0.5
    return float(np.clip((d - d_min) / span, 0.0, 1.0))


def _psb_eye_branch_scalars(
    t: float,
    *,
    y_left: float,
    y_right: float,
    y_s_ref: float,
    y_t_ref: float,
    merge_epsilon_sep: float = 1.5e-8,
) -> tuple[float, float]:
    """Singlet / triplet readout scalars: smooth PSB eye **bounded** by base states.

    ``y_s_ref`` / ``y_t_ref`` are the most-separated singlet and triplet means
    (full PSB contrast). The merged trajectory ``merge(t) = ramp(y_left→y_right)``
    must stay inside ``[y_s_ref, y_t_ref]``. Opening uses ``bump = sin^2(pi t)``:

        y_S = (1-bump)·merge + bump·y_s_ref
        y_T = (1-bump)·merge + bump·y_t_ref

    so at the sweep ends (bump→0) both states sit on ``merge``, and at the
    center (bump→1) they sit exactly on the base states—never beyond them.
    """
    if y_t_ref <= y_s_ref:
        raise ValueError("require y_t_ref > y_s_ref")
    bump = float(np.sin(np.pi * t) ** 2)
    merge = y_left * (1.0 - t) + y_right * t
    if bump < 1e-14:
        h = 0.5 * merge_epsilon_sep
        return merge - h, merge + h
    y_singlet = (1.0 - bump) * merge + bump * y_s_ref
    y_triplet = (1.0 - bump) * merge + bump * y_t_ref
    return y_singlet, y_triplet


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _global_readout_axis_from_endpoints(
    I: np.ndarray, Q: np.ndarray, *, n_edge: int
) -> tuple[float, float]:
    """Unit vector (c, s) in the I–Q plane from low-detuning vs high-detuning centroids."""
    n_det = I.shape[1]
    k = max(1, min(n_edge, n_det // 2 or 1))
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
    sweep_name: str = "detuning",
    qubit_pairs: Optional[Sequence[Union[str, object]]] = None,
    n_bins: int = 48,
    log_counts: bool = True,
    edge_fraction: float = 1.0 / 3.0,
) -> "Figure":
    """2D shot-density map vs sweep **without** running Barthel / PCA fits.

    A single readout axis per pair is inferred from the difference between mean
    IQ on the low-``sweep_name`` edge vs the high edge (same dataset only).

    Parameters
    ----------
    ds
        Dataset with ``I``, ``Q`` and dims including ``qubit_pair``, ``n_runs``,
        and ``sweep_name``.
    qubit_pairs
        Names or objects with a ``.name`` attribute to plot; default is all
        values on the ``qubit_pair`` coordinate.
    edge_fraction
        Fraction of sweep points (rounded up) used at each end to form centroids.
    """
    import matplotlib.pyplot as plt

    if sweep_name not in ds.dims and sweep_name not in ds.coords:
        raise KeyError(f"sweep_name={sweep_name!r} not found on ds")

    if qubit_pairs is None:
        names = [str(x) for x in ds["qubit_pair"].values]
    else:
        names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]

    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    sweep_vals = np.asarray(ds[sweep_name].values, dtype=float)
    attrs = ds[sweep_name].attrs if sweep_name in ds.coords else {}
    long_name = attrs.get("long_name", sweep_name)
    units = attrs.get("units")
    x_label = f"{long_name} [{units}]" if units else long_name

    n_det = len(sweep_vals)
    n_edge = max(1, int(np.ceil(n_det * edge_fraction)))

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        I = np.asarray(ds.I.sel(qubit_pair=name).values, dtype=float)
        Q = np.asarray(ds.Q.sel(qubit_pair=name).values, dtype=float)
        if I.ndim != 2 or Q.shape != I.shape:
            raise ValueError(f"Expected I,Q 2D (n_runs, {sweep_name}); got {I.shape}")

        c, s = _global_readout_axis_from_endpoints(I, Q, n_edge=n_edge)
        proj_cols = [I[:, si] * c + Q[:, si] * s for si in range(n_det)]
        flat = np.concatenate(proj_cols) if proj_cols else np.array([0.0])
        lo, hi = np.nanpercentile(flat, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        y_min, y_max = lo - pad, hi + pad
        edges = np.linspace(y_min, y_max, n_bins + 1)

        H = np.zeros((n_det, n_bins), dtype=float)
        for si in range(n_det):
            H[si, :], _ = np.histogram(proj_cols[si], bins=edges, density=False)

        Z = np.log10(H.T + 1.0) if log_counts else H.T

        if n_det == 1:
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
    """Synthetic PSB detuning sweep: Barthel-style IQ via :func:`simulate_readout_iq`.

    Singlet and triplet **means move with detuning** along smooth branches that
    merge at the sweep ends and approach fixed base levels ``y_s_ref``, ``y_t_ref``
    at the center (``sin^2`` opening); means never lie outside those base states.
    Shot noise and T→S relaxation during readout use the same forward model as
    elsewhere in the stack.

    To inspect raw data without running Barthel analysis, use
    :func:`plot_simulated_dataset_histograms`.
    """
    qubit_pairs = _resolve_qubit_pairs(node)
    pair_names = [qp.name for qp in qubit_pairs]

    detuning_min = float(node.parameters.detuning_min)
    detuning_max = float(node.parameters.detuning_max)
    detuning_points = int(node.parameters.detuning_points)
    detuning_array = np.linspace(detuning_min, detuning_max, detuning_points)

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(pair_names),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "voltage", "units": "V"}),
    }

    num_shots = int(node.parameters.num_shots)
    n_pairs = len(qubit_pairs)
    n_det = len(detuning_array)

    I_arr = np.zeros((n_pairs, num_shots, n_det), dtype=float)
    Q_arr = np.zeros((n_pairs, num_shots, n_det), dtype=float)

    tau_M = 1.0
    T1 = 2.0
    sigma_I = 0.12e-2
    sigma_Q = 0.10e-2

    for pi, _qp in enumerate(qubit_pairs):
        rng = np.random.default_rng(seed=42 + pi * 9973)
        # Readout axis in the I–Q plane (per logical pair)
        theta = 0.38 + 0.27 * float(pi)
        c_ax, s_ax = float(np.cos(theta)), float(np.sin(theta))
        # Base S/T readout bounds (max PSB separation); merged ends stay strictly inside.
        y_s_ref = (-1.35e-2) * (1.0 + 0.06 * float(pi))
        y_t_ref = (1.35e-2) * (1.0 + 0.06 * float(pi))
        y_left = float(np.clip((-1.15e-2) * (1.0 + 0.12 * float(pi)), y_s_ref + 1e-12, y_t_ref - 1e-12))
        y_right = float(np.clip((1.05e-2) * (1.0 + 0.12 * float(pi)), y_s_ref + 1e-12, y_t_ref - 1e-12))

        for di, d in enumerate(detuning_array):
            t = _detuning_unit(d, detuning_min, detuning_max)
            y_s, y_t = _psb_eye_branch_scalars(
                t,
                y_left=y_left,
                y_right=y_right,
                y_s_ref=y_s_ref,
                y_t_ref=y_t_ref,
            )
            mu_s = (y_s * c_ax, y_s * s_ax)
            mu_t = (y_t * c_ax, y_t * s_ax)
            # Balanced mixture in the split region; when means merge, label is irrelevant.
            params = SimulationParamsIQ(
                n_samples=num_shots,
                p_triplet=0.5,
                mu_S=mu_s,
                mu_T=mu_t,
                sigma_I=sigma_I,
                sigma_Q=sigma_Q,
                rho=0.0,
                tau_M=tau_M,
                T1=T1,
            )
            X, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
            I_arr[pi, :, di] = X[:, 0]
            Q_arr[pi, :, di] = X[:, 1]

    return xr.Dataset(
        {
            "I": (["qubit_pair", "n_runs", "detuning"], I_arr),
            "Q": (["qubit_pair", "n_runs", "detuning"], Q_arr),
        },
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            "detuning": xr.DataArray(
                detuning_array, dims="detuning", attrs={"long_name": "voltage", "units": "V"}
            ),
        },
    )
