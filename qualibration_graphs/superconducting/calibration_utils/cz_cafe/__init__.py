"""Context Aware Fidelity Estimation (CAFE) utilities for the CZ gate (arXiv:2303.17565)."""

from .analysis import (
    FitResults,
    QuadraticBudget,
    fit_cafe_curve,
    fit_raw_data,
    log_fitted_results,
    model_fidelity,
    primary_variant,
    process_raw_dataset,
)
from .circuits import (
    NUM_ANGLES_PER_PART,
    NUM_STATES,
    CafeCircuitAngles,
    build_circuit_angles,
    reference_gates_per_pair,
)
from .parameters import Parameters
from .plotting import plot_leakage, plot_raw_data_with_fit

__all__ = [
    "CafeCircuitAngles",
    "FitResults",
    "NUM_ANGLES_PER_PART",
    "NUM_STATES",
    "Parameters",
    "QuadraticBudget",
    "build_circuit_angles",
    "fit_cafe_curve",
    "fit_raw_data",
    "log_fitted_results",
    "model_fidelity",
    "plot_leakage",
    "plot_raw_data_with_fit",
    "primary_variant",
    "process_raw_dataset",
    "reference_gates_per_pair",
]
