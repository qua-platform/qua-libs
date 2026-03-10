from .base_parameters import GateVirtualizationBaseParameters, get_voltage_arrays
from .sensor_dot_tuning_parameters import SensorDotTuningParameters
from .sensor_compensation_parameters import SensorCompensationParameters
from .virtual_plunger_parameters import VirtualPlungerParameters
from .barrier_pat_parameters import PATLeverArmParameters, BarrierCompensationParameters
from .analysis import process_raw_dataset, update_compensation_matrix, update_compensation_submatrix
from .sensor_dot_analysis import fit_lorentzian, lorentzian, optimal_operating_point
from .sensor_compensation_analysis import (
    extract_sensor_compensation_coefficients,
    fit_shifted_lorentzian,
    shifted_lorentzian_2d,
)
from .virtual_plunger_analysis import extract_virtual_plunger_coefficients
from .barrier_compensation_analysis import (
    extract_barrier_compensation_coefficients,
    evaluate_slope_fit_acceptance,
    resolve_pair_calibration_topology,
)
from .plotting import (
    plot_2d_scan,
    plot_barrier_pair_diagnostics,
    plot_barrier_transform_history,
    plot_compensation_fit,
    plot_detuning_fit_family,
    plot_sensor_compensation_diagnostic,
    plot_target_barrier_coupling_summary,
    plot_tunnel_slope_fit,
    plot_virtual_gate_matrix,
    plot_virtual_plunger_diagnostic,
)


def create_2d_scan_program(*args, **kwargs):
    from .scan_utils import create_2d_scan_program as _create_2d_scan_program

    return _create_2d_scan_program(*args, **kwargs)


def setup_qdac_dc_lists(*args, **kwargs):
    from .scan_utils import setup_qdac_dc_lists as _setup_qdac_dc_lists

    return _setup_qdac_dc_lists(*args, **kwargs)


def read_qdac_voltage(*args, **kwargs):
    from .scan_utils import _read_qdac_voltage as _read_qdac_voltage_impl

    return _read_qdac_voltage_impl(*args, **kwargs)


__all__ = [
    "GateVirtualizationBaseParameters",
    "SensorDotTuningParameters",
    "SensorCompensationParameters",
    "VirtualPlungerParameters",
    "PATLeverArmParameters",
    "BarrierCompensationParameters",
    "get_voltage_arrays",
    "create_2d_scan_program",
    "setup_qdac_dc_lists",
    "read_qdac_voltage",
    "process_raw_dataset",
    "update_compensation_matrix",
    "update_compensation_submatrix",
    "fit_lorentzian",
    "lorentzian",
    "optimal_operating_point",
    "extract_sensor_compensation_coefficients",
    "fit_shifted_lorentzian",
    "shifted_lorentzian_2d",
    "extract_virtual_plunger_coefficients",
    "extract_barrier_compensation_coefficients",
    "evaluate_slope_fit_acceptance",
    "resolve_pair_calibration_topology",
    "plot_sensor_compensation_diagnostic",
    "plot_2d_scan",
    "plot_compensation_fit",
    "plot_virtual_plunger_diagnostic",
    "plot_detuning_fit_family",
    "plot_barrier_pair_diagnostics",
    "plot_target_barrier_coupling_summary",
    "plot_tunnel_slope_fit",
    "plot_barrier_transform_history",
    "plot_virtual_gate_matrix",
]
