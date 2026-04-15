"""
readout_barthel
A small library to simulate Barthel-style single-shot readout traces for a two-state system
with decay during measurement, and to fit flexible probabilistic models with NumPyro.
"""

# Simulation
from .simulate import (
    SimulationParams,
    simulate_readout,
    SimulationParamsIQ,
    simulate_readout_iq,
)

# Calibration
from .calibrate import (
    CalibrationResult,
    Barthel1DFromIQ,
)

# Classification
from .classify import classify_iq_with_pca_threshold

# PCA utilities
from .pca import (
    PCAProjection,
    pca_project_1d,
)

# Fitting
from .fit import (
    MCMCConfig,
    fit_model,
)

# Utilities
from .utils import (
    Normalizer1D,
    Barthel1DMetricCurves,
)

# Plotting
from .plotting import (
    plot_raw_data,
    plot_fit,
    plot_raw_data_iq,
    plot_fit_iq,
    plot_barthel_fit_1d,
    plot_fidelity_and_visibility_barthel_1d,
    plot_iq_with_pca_and_threshold,
)

# Analytic functions
from .analytic import (
    decay_inflight_integral,
    triplet_pdf_analytic,
    triplet_cdf_analytic,
)

# Models
from .models.barthel_model import build_barthel_model_1d_analytic
from .models.gmm_model import (
    make_gmm_model_factory,
    build_gmm_model,
    make_gmm_model_factory_2d,
    compute_bic,
    compute_bic_2d_diag,
)

__all__ = [
    # Simulation
    "SimulationParams",
    "simulate_readout",
    "SimulationParamsIQ",
    "simulate_readout_iq",
    # Calibration
    "CalibrationResult",
    "Barthel1DFromIQ",
    # Classification
    "classify_iq_with_pca_threshold",
    # PCA
    "PCAProjection",
    "pca_project_1d",
    # Fitting
    "MCMCConfig",
    "fit_model",
    # Utilities
    "Normalizer1D",
    "Barthel1DMetricCurves",
    # Plotting
    "plot_raw_data",
    "plot_fit",
    "plot_raw_data_iq",
    "plot_fit_iq",
    "plot_barthel_fit_1d",
    "plot_fidelity_and_visibility_barthel_1d",
    "plot_iq_with_pca_and_threshold",
    # Analytic
    "decay_inflight_integral",
    "triplet_pdf_analytic",
    "triplet_cdf_analytic",
    # Models
    "build_barthel_model_1d_analytic",
    "make_gmm_model_factory",
    "build_gmm_model",
    "make_gmm_model_factory_2d",
    "compute_bic",
    "compute_bic_2d_diag",
]
