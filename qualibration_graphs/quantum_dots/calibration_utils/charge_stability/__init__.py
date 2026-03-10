from .parameters import (
    NodeSpecificParameters,
    OPXQDACParameters,
    Parameters,
    get_voltage_arrays,
    prepare_dc_lists,
)
from .plotting import (
    plot_raw_amplitude,
    plot_raw_phase,
    plot_individual_raw_amplitude,
    plot_individual_raw_phase,
    plot_change_point_overlays,
    plot_line_fit_overlays,
)
from .analysis import (
    process_raw_dataset,
    fit_raw_data,
    fit_individual_raw_data,
    log_fitted_results,
    FitParameters,
)

from .scan_modes import (
    ScanMode,
    RasterScan,
    SwitchRasterScan,
)

try:
    from .edge_line_analysis import analyze_edge_map, SegmentFit
except ImportError:  # pragma: no cover - optional dependency guard
    analyze_edge_map = None
    SegmentFit = None

OPXParameters = Parameters
OPXQDACParameters = Parameters

__all__ = [
    "Parameters",
    "NodeSpecificParameters",
    "OPXParameters",
    "OPXQDACParameters",
    "get_voltage_arrays",
    "prepare_dc_lists",
    "plot_raw_amplitude",
    "plot_raw_phase",
    "plot_individual_raw_amplitude",
    "plot_individual_raw_phase",
    "plot_change_point_overlays",
    "plot_line_fit_overlays",
    "process_raw_dataset",
    "fit_raw_data",
    "fit_individual_raw_data",
    "log_fitted_results",
    "FitParameters",
    "analyze_edge_map",
    "SegmentFit",
    "ScanMode",
    "RasterScan",
    "SwitchRasterScan",
]
