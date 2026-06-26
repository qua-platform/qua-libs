from __future__ import annotations

import numpy as np
import xarray as xr


def _pc1_std(i_vals: np.ndarray, q_vals: np.ndarray) -> float:
    """Return the standard deviation along the first principal component."""
    points = np.column_stack([i_vals, q_vals]).astype(np.float64)
    points -= points.mean(axis=0, keepdims=True)
    cov = np.cov(points, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(float(eigvals[-1]), 0.0)))


def analyse_detuning_vs_buffer(ds_raw: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """Compute slim PCA-like contrast maps for detuning-vs-buffer sweeps."""
    i_data = ds_raw["I"].values
    q_data = ds_raw["Q"].values

    n_pairs, _, n_detuning, n_buffer = i_data.shape
    pc1_map = np.zeros((n_pairs, n_detuning, n_buffer), dtype=np.float64)
    iq_trace_map = np.zeros((n_pairs, n_detuning, n_buffer), dtype=np.float64)

    for p_idx in range(n_pairs):
        for d_idx in range(n_detuning):
            for b_idx in range(n_buffer):
                i_vals = i_data[p_idx, :, d_idx, b_idx]
                q_vals = q_data[p_idx, :, d_idx, b_idx]
                pc1_map[p_idx, d_idx, b_idx] = _pc1_std(i_vals, q_vals)
                iq_trace_map[p_idx, d_idx, b_idx] = float(
                    np.trace(np.cov(np.column_stack([i_vals, q_vals]).T))
                )

    ds_fit = xr.Dataset(
        data_vars={
            "pc1_std": xr.DataArray(
                pc1_map,
                dims=["qubit_pair", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": ds_raw["qubit_pair"].values,
                    "detuning": ds_raw["detuning"].values,
                    "buffer_duration": ds_raw["buffer_duration"].values,
                },
                attrs={
                    "long_name": "PC1 spread",
                    "units": "arb.",
                },
            ),
            "iq_trace": xr.DataArray(
                iq_trace_map,
                dims=["qubit_pair", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": ds_raw["qubit_pair"].values,
                    "detuning": ds_raw["detuning"].values,
                    "buffer_duration": ds_raw["buffer_duration"].values,
                },
                attrs={
                    "long_name": "I/Q covariance trace",
                    "units": "arb.",
                },
            ),
        }
    )

    fit_results = {}
    for pair_name in ds_raw["qubit_pair"].values:
        metric = ds_fit["pc1_std"].sel(qubit_pair=pair_name).values
        best_flat = int(np.argmax(metric))
        d_idx, b_idx = np.unravel_index(best_flat, metric.shape)
        fit_results[str(pair_name)] = {
            "success": True,
            "optimal_detuning": float(ds_raw["detuning"].values[d_idx]),
            "optimal_buffer_duration": int(ds_raw["buffer_duration"].values[b_idx]),
            "max_pc1_std": float(metric[d_idx, b_idx]),
        }

    return ds_fit, fit_results
