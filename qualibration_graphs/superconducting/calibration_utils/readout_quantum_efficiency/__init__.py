"""Public API for the readout quantum efficiency calibration utilities."""

from .parameters import Parameters
from .analysis import process_raw_dataset, fit_raw_data, log_fitted_results
from .plotting import (
    plot_efficiency_vs_frequency,
    plot_efficiency_map,
    plot_snr_and_dephasing,
    plot_raw_fringes,
    plot_fringe_amplitude_vs_power,
    plot_stark_phase,
)

__all__ = [
    "Parameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "plot_efficiency_vs_frequency",
    "plot_efficiency_map",
    "plot_snr_and_dephasing",
    "plot_raw_fringes",
    "plot_fringe_amplitude_vs_power",
    "plot_stark_phase",
]
