from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)
from calibration_utils.psb_search_sweep_detuning.simulated_data_generator import (
    _psb_eye_branch_scalars,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qualibrate.core import QualibrationNode


def _resolve_qubit_pairs(node: "QualibrationNode") -> List:
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def canonicalize_fixed_point_ds_raw(ds: xr.Dataset, sweep_name: str = "singleton") -> xr.Dataset:
    """Force ``I`` and ``Q`` to dims ``(qubit_pair, n_runs)`` — same layout for hardware and simulation.

    Drops a length-1 ``sweep_name`` dimension (or any other stray size-1 axis) so ``XarrayDataFetcher``
    metadata and simulated data match.
    """
    ds_out = ds.copy()
    for key in ("I", "Q"):
        if key not in ds_out.data_vars:
            continue
        da = ds_out[key]
        if sweep_name in da.dims and da.sizes.get(sweep_name, 0) == 1:
            da = da.isel({sweep_name: 0}, drop=True)
        for d in list(da.dims):
            if d not in ("qubit_pair", "n_runs") and da.sizes.get(d, 0) == 1:
                da = da.squeeze(d, drop=True)
        for need in ("qubit_pair", "n_runs"):
            if need not in da.dims:
                raise ValueError(
                    f"canonicalize_fixed_point_ds_raw: {key!r} missing dim {need!r} after squeeze; dims={da.dims}"
                )
        ds_out[key] = da.transpose("qubit_pair", "n_runs")
    return ds_out


def generate_simulated_dataset(node: "QualibrationNode") -> xr.Dataset:
    """Synthetic single-point PSB IQ: one histogram per pair, ~50/50 S/T via Barthel model.

    Uses the same branch geometry as the detuning sweep simulator at ``t = 0.5`` (mid-opening
    PSB eye) so singlet/triplet means are well separated in the projected readout direction.
    :class:`~calibration_utils.iq_blobs.readout_barthel.simulate.SimulationParamsIQ` uses
    ``p_triplet = 0.5`` for a balanced mixture.
    """
    qubit_pairs = _resolve_qubit_pairs(node)
    pair_names = [qp.name for qp in qubit_pairs]

    sweep_name = node.parameters.sweep_name

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = [qp.quantum_dot_pair for qp in qubit_pairs]
    # Match hardware ``create_qua_program``: only axes that correspond to streamed buffers (no dummy sweep dim).
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(pair_names),
        "n_runs": xr.DataArray(np.arange(node.parameters.num_shots), attrs={"long_name": "shot"}),
    }

    num_shots = int(node.parameters.num_shots)
    n_pairs = len(qubit_pairs)

    I_arr = np.zeros((n_pairs, num_shots), dtype=float)
    Q_arr = np.zeros((n_pairs, num_shots), dtype=float)

    tau_M = 1.0
    T1 = 2.0
    sigma_I = 0.12e-2
    sigma_Q = 0.10e-2
    t_mid = 0.5

    for pi, _qp in enumerate(qubit_pairs):
        rng = np.random.default_rng(seed=42_001 + pi * 9973)
        theta = 0.38 + 0.27 * float(pi)
        c_ax, s_ax = float(np.cos(theta)), float(np.sin(theta))
        y_s_ref = (-0.5e-2) * (1.0 + 0.06 * float(pi))
        y_t_ref = (0.5e-2) * (1.0 + 0.06 * float(pi))
        y_left = float(np.clip((-1.15e-2) * (1.0 + 0.12 * float(pi)), y_s_ref + 1e-12, y_t_ref - 1e-12))
        y_right = float(np.clip((1.05e-2) * (1.0 + 0.12 * float(pi)), y_s_ref + 1e-12, y_t_ref - 1e-12))

        y_s, y_t = _psb_eye_branch_scalars(
            t_mid,
            y_left=y_left,
            y_right=y_right,
            y_s_ref=y_s_ref,
            y_t_ref=y_t_ref,
        )
        mu_s = (y_s * c_ax, y_s * s_ax)
        mu_t = (y_t * c_ax, y_t * s_ax)

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
        I_arr[pi, :] = X[:, 0]
        Q_arr[pi, :] = X[:, 1]

    ds = xr.Dataset(
        {
            "I": (["qubit_pair", "n_runs"], I_arr),
            "Q": (["qubit_pair", "n_runs"], Q_arr),
        },
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
        },
    )
    # Scalar metadata (not a dimension) so ``sweep_name`` can still be referenced on ``ds``.
    return ds.assign_coords({sweep_name: float(0.0)})


def _grid_subplots(n: int) -> tuple[int, int]:
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _project_shots_pca1d(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-shot scalar along the first principal axis of the (I, Q) cloud (mean-centred)."""
    m = np.column_stack([I.ravel(), Q.ravel()]).astype(np.float64)
    if m.shape[0] < 2 or not np.all(np.isfinite(m)):
        return I.ravel().astype(np.float64)
    m = m - np.nanmean(m, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    w = vt[0]
    nrm = float(np.hypot(w[0], w[1]))
    if nrm < 1e-20:
        return m[:, 0]
    w = w / nrm
    return (m @ w).ravel()


def plot_simulated_histograms(
    ds: xr.Dataset,
    *,
    sweep_name: str = "singleton",
    qubit_pairs: Optional[Sequence[Union[str, object]]] = None,
    n_bins: int = 48,
) -> "Figure":
    """1D histogram per pair: **counts** vs **bin centre** (projected readout from shot PCA on I/Q)."""
    if sweep_name not in ds.coords and sweep_name not in ds.dims:
        raise KeyError(f"sweep_name={sweep_name!r} not found on ds (coords or dims)")

    if qubit_pairs is None:
        names = [str(x) for x in ds["qubit_pair"].values]
    else:
        names = [p if isinstance(p, str) else getattr(p, "name", str(p)) for p in qubit_pairs]

    if sweep_name in ds.dims:
        sweep_vals = np.asarray(ds[sweep_name].values, dtype=float)
        if len(sweep_vals) != 1:
            raise ValueError(
                f"plot_simulated_histograms expects a single {sweep_name!r} slice; got {len(sweep_vals)} values."
            )

    n = len(names)
    n_rows, n_cols = _grid_subplots(n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        I = np.asarray(ds.I.sel(qubit_pair=name).values, dtype=float).ravel()
        Q = np.asarray(ds.Q.sel(qubit_pair=name).values, dtype=float).ravel()
        proj = _project_shots_pca1d(I, Q)
        counts, edges = np.histogram(proj, bins=n_bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        ax.bar(centres, counts, width=widths, align="center", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Projected readout (1st PC of I/Q)")
        ax.set_ylabel("Counts")
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Simulated readout: histogram (counts vs bins)")
    fig.tight_layout()
    return fig
