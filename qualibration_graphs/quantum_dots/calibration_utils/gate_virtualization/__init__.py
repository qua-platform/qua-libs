from .parameters import (
    GateVirtualizationBaseParameters,
    SensorCompensationParameters,
    VirtualPlungerParameters,
    BarrierCompensationParameters,
    get_voltage_arrays,
)
from .analysis import process_raw_dataset, update_compensation_matrix
from .sensor_compensation_analysis import (
    extract_sensor_compensation_coefficients,
    fit_shifted_lorentzian,
    shifted_lorentzian_2d,
)
from .virtual_plunger_analysis import extract_virtual_plunger_coefficients
from .barrier_compensation_analysis import extract_barrier_compensation_coefficients
from .plotting import (
    plot_sensor_compensation_diagnostic,
    plot_2d_scan,
    plot_compensation_fit,
)


def read_qdac_voltage(*args, **kwargs):
    from .scan_utils import _read_qdac_voltage as _read_qdac_voltage_impl

    return _read_qdac_voltage_impl(*args, **kwargs)


def create_2d_scan_program(*args, **kwargs):
    from .scan_utils import create_2d_scan_program as _create_2d_scan_program

    return _create_2d_scan_program(*args, **kwargs)


def setup_qdac_dc_lists(*args, **kwargs):
    from .scan_utils import setup_qdac_dc_lists as _setup_qdac_dc_lists

    return _setup_qdac_dc_lists(*args, **kwargs)


__all__ = [
    "GateVirtualizationBaseParameters",
    "SensorCompensationParameters",
    "VirtualPlungerParameters",
    "BarrierCompensationParameters",
    "get_voltage_arrays",
    "create_2d_scan_program",
    "setup_qdac_dc_lists",
    "read_qdac_voltage",
    "process_raw_dataset",
    "update_compensation_matrix",
    "extract_sensor_compensation_coefficients",
    "fit_shifted_lorentzian",
    "shifted_lorentzian_2d",
    "extract_virtual_plunger_coefficients",
    "extract_barrier_compensation_coefficients",
    "plot_sensor_compensation_diagnostic",
    "plot_2d_scan",
    "plot_compensation_fit",
]
