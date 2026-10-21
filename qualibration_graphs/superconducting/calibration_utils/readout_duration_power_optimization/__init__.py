"""Public API for the joint readout duration x power optimization utilities."""

from .parameters import (
    Parameters,
    get_amplitude_prefactors,
    get_chunk_duration_in_ns,
    get_durations_in_ns,
    get_samples_per_chunk,
)
from .analysis import (
    FitParameters,
    OperatingPoint,
    blob_statistics,
    fit_raw_data,
    log_fitted_results,
    process_raw_dataset,
    select_operating_point,
    to_volts_per_duration,
)
from .plotting import plot_amplitude_cut, plot_duration_cut, plot_fidelity_map
from .qua_sequence import (
    DEFAULT_INTEGRATION_WEIGHTS,
    declare_recombination_variables,
    has_custom_integration_weights,
    raw_integration_weights,
    readout_config_override,
    save_accumulated_quadratures,
    set_integration_weights,
)

__all__ = [
    "DEFAULT_INTEGRATION_WEIGHTS",
    "Parameters",
    "get_amplitude_prefactors",
    "get_chunk_duration_in_ns",
    "get_durations_in_ns",
    "get_samples_per_chunk",
    "FitParameters",
    "OperatingPoint",
    "blob_statistics",
    "fit_raw_data",
    "log_fitted_results",
    "process_raw_dataset",
    "select_operating_point",
    "to_volts_per_duration",
    "plot_amplitude_cut",
    "plot_duration_cut",
    "plot_fidelity_map",
    "declare_recombination_variables",
    "has_custom_integration_weights",
    "raw_integration_weights",
    "readout_config_override",
    "save_accumulated_quadratures",
    "set_integration_weights",
]
