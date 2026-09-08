"""
Intentionally re-uses the shared `iq_sweep` analysis implementation
without modifying it, so that the 06x node family stays consistent.
"""

from calibration_utils.iq_utils import (
    FitParameters,
    fit_raw_data,
    fit_raw_data_pca_gaussian,
    log_fitted_results,
    process_raw_dataset,
)

__all__ = [
    "FitParameters",
    "process_raw_dataset",
    "fit_raw_data",
    "fit_raw_data_pca_gaussian",
    "log_fitted_results",
    "extract_vgs_id",
]


def extract_vgs_id(qubit_pairs):
    vgs_id = next(iter({pair.quantum_dot_pair.voltage_sequence.gate_set.name for pair in qubit_pairs}))
    return vgs_id
