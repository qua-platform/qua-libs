from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from calibration_utils.iq_utils.analysis import process_raw_dataset as process_iq_raw_dataset

__all__ = [
    "FitParameters",
    "analyse_detuning_vs_buffer",
    "fit_detuning_vs_buffer_raw_data",
    "log_fitted_results",
    "process_raw_dataset",
]


@dataclass
class FitParameters:
    """Per-pair summary of the exploratory detuning-vs-buffer analysis."""

    metric_name: str
    optimal_detuning: float
    optimal_detuning_index: int
    optimal_buffer_duration: int
    optimal_buffer_duration_index: int
    max_metric_value: float
    max_pc1_std: float
    max_iq_trace: float
    success: bool


def process_raw_dataset(ds_raw: xr.Dataset, *_args, **_kwargs) -> xr.Dataset:
    """Process raw dataset into an analysis-ready form (keeps ``ds_raw`` immutable)."""
    return process_iq_raw_dataset(ds_raw)


def _pc1_std(i_vals: np.ndarray, q_vals: np.ndarray) -> float:
    """Return the standard deviation along the first principal component."""
    points = np.column_stack([i_vals, q_vals]).astype(np.float64)
    points -= points.mean(axis=0, keepdims=True)
    cov = np.cov(points, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(float(eigvals[-1]), 0.0)))


def analyse_detuning_vs_buffer(ds_raw: xr.Dataset) -> xr.Dataset:
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
                iq_trace_map[p_idx, d_idx, b_idx] = float(np.trace(np.cov(np.column_stack([i_vals, q_vals]).T)))

    return xr.Dataset(
        data_vars={
            "pc1_std": xr.DataArray(
                pc1_map,
                dims=["qubit_pair", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": ds_raw["qubit_pair"].values,
                    "detuning": ds_raw["detuning"].values,
                    "buffer_duration": ds_raw["buffer_duration"].values,
                },
                attrs={"long_name": "PC1 spread", "units": "arb."},
            ),
            "iq_trace": xr.DataArray(
                iq_trace_map,
                dims=["qubit_pair", "detuning", "buffer_duration"],
                coords={
                    "qubit_pair": ds_raw["qubit_pair"].values,
                    "detuning": ds_raw["detuning"].values,
                    "buffer_duration": ds_raw["buffer_duration"].values,
                },
                attrs={"long_name": "I/Q covariance trace", "units": "arb."},
            ),
        }
    )


def fit_detuning_vs_buffer_raw_data(node) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Analyse the processed dataset and extract the best 2D operating point per pair."""
    ds_processed = node.results["ds_processed"]
    metric_name = node.parameters.pca_metric
    ds_fit = analyse_detuning_vs_buffer(ds_processed)

    fit_results: dict[str, FitParameters] = {}
    for pair_name in ds_fit["qubit_pair"].values:
        metric = np.asarray(ds_fit[metric_name].sel(qubit_pair=pair_name).values, dtype=float)
        if metric.size == 0 or not np.any(np.isfinite(metric)):
            fit_results[str(pair_name)] = FitParameters(
                metric_name=metric_name,
                optimal_detuning=float("nan"),
                optimal_detuning_index=-1,
                optimal_buffer_duration=-1,
                optimal_buffer_duration_index=-1,
                max_metric_value=float("nan"),
                max_pc1_std=float("nan"),
                max_iq_trace=float("nan"),
                success=False,
            )
            continue

        best_flat = int(np.nanargmax(metric))
        d_idx, b_idx = np.unravel_index(best_flat, metric.shape)
        max_metric_value = float(metric[d_idx, b_idx])
        success = bool(np.isfinite(max_metric_value) and max_metric_value > 0.0)

        fit_results[str(pair_name)] = FitParameters(
            metric_name=metric_name,
            optimal_detuning=float(ds_fit["detuning"].values[d_idx]),
            optimal_detuning_index=int(d_idx),
            optimal_buffer_duration=int(ds_fit["buffer_duration"].values[b_idx]),
            optimal_buffer_duration_index=int(b_idx),
            max_metric_value=max_metric_value,
            max_pc1_std=float(np.nanmax(ds_fit["pc1_std"].sel(qubit_pair=pair_name).values)),
            max_iq_trace=float(np.nanmax(ds_fit["iq_trace"].sel(qubit_pair=pair_name).values)),
            success=success,
        )

    return ds_fit, fit_results


def log_fitted_results(fit_results: dict, log_callable=None):
    """Log the selected 2D operating point for each qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qubit_pair, result in fit_results.items():
        status = " SUCCESS!" if result["success"] else " FAIL!"
        summary = (
            f"Results for qubit pair {qubit_pair}:{status}\n"
            f"optimal detuning = {result['optimal_detuning']:.4g} V | "
            f"optimal buffer_duration = {result['optimal_buffer_duration']} ns | "
            f"{result['metric_name']} = {result['max_metric_value']:.4g} | "
            f"max pc1_std = {result['max_pc1_std']:.4g} | "
            f"max iq_trace = {result['max_iq_trace']:.4g}"
        )
        log_callable(summary)
