"""G-E-F readout duration optimization utilities (node 14c)."""

from .parameters import Parameters, get_durations_in_ns, get_samples_per_chunk
from .analysis import (
    DurationChoice,
    FitParameters,
    fit_raw_data,
    log_fitted_results,
    process_raw_dataset,
    select_duration,
)
from .plotting import plot_fidelity_vs_duration

__all__ = [
    "Parameters",
    "get_durations_in_ns",
    "get_samples_per_chunk",
    "DurationChoice",
    "FitParameters",
    "fit_raw_data",
    "log_fitted_results",
    "process_raw_dataset",
    "select_duration",
    "plot_fidelity_vs_duration",
]
