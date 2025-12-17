from .parameters import Parameters, get_voltage_arrays, get_swept_object
from .plotting import (
    plot_raw_amplitude,
    plot_raw_phase,
    plot_individual_raw_amplitude,
    plot_individual_raw_phase,
    plot_change_point_overlays,
    plot_peak_locations,
    plot_gap_shoulders,
    plot_charge_state_boundaries,
    plot_line_fit_overlays,
)
from .analysis import (
    process_raw_dataset,
    fit_raw_data,
    fit_individual_raw_data,
    log_fitted_results,
    FitParameters,
)
try:
    from .edge_line_analysis import analyze_edge_map, SegmentFit
except ImportError:  # pragma: no cover - optional dependency guard
    analyze_edge_map = None
    SegmentFit = None

__all__ = [
    "Parameters",
    "get_voltage_arrays",
    "get_swept_object",
    "plot_raw_amplitude",
    "plot_raw_phase",
    "plot_individual_raw_amplitude",
    "plot_individual_raw_phase",
    "plot_change_point_overlays",
    "plot_peak_locations",
    "plot_gap_shoulders",
    "plot_charge_state_boundaries",
    "plot_line_fit_overlays",
    "process_raw_dataset",
    "fit_raw_data",
    "fit_individual_raw_data",
    "log_fitted_results",
    "FitParameters",
    "analyze_edge_map",
    "SegmentFit",
]
