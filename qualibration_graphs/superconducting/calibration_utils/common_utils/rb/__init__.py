"""Randomized benchmarking calculations shared by the single- and two-qubit RB nodes.

Survival decay fit and checks (``survival``), alpha -> fidelity for Standard and Interleaved RB
(``fidelity``), leakage model, fit and per-gate subtraction (``leakage``), and g/e/f population
correction and statistics (``populations``), the leakage pre-run check (``checks``), and loading the
saved Standard RB run an interleaved node compares against (``reference``). Validation, logging and plotting stay in each RB
node's own calibration_utils package.
"""

from .checks import validate_leakage_run
from .fidelity import (
    IRBFidelity,
    SRBFidelity,
    as_ufloat,
    clifford_fidelity_from_alpha,
    compute_irb_fidelity,
    compute_srb_fidelity,
    interleaved_gate_fidelity_from_alpha,
    unwrap,
    validate_fidelity_bounds,
    validate_fidelity_uncertainty,
    validate_interleaved_alpha,
)
from .leakage import (
    SHALLOW_LEAKAGE_THRESHOLD,
    LeakageFit,
    fit_leakage,
    leakage_decay_curve,
    leakage_per_gate,
)
from .reference import load_saved_ds_fit
from .populations import correct_populations, mean_and_stderr_over_sequences, read_confusion_matrix
from .survival import (
    RESIDUAL_SIGMA_LIMIT,
    fit_survival,
    rb_decay_curve,
    validate_fit_parameters,
    validate_residuals,
)

__all__ = [
    "IRBFidelity",
    "SRBFidelity",
    "as_ufloat",
    "clifford_fidelity_from_alpha",
    "compute_irb_fidelity",
    "compute_srb_fidelity",
    "interleaved_gate_fidelity_from_alpha",
    "unwrap",
    "validate_fidelity_bounds",
    "validate_fidelity_uncertainty",
    "validate_interleaved_alpha",
    "SHALLOW_LEAKAGE_THRESHOLD",
    "LeakageFit",
    "fit_leakage",
    "leakage_decay_curve",
    "leakage_per_gate",
    "correct_populations",
    "mean_and_stderr_over_sequences",
    "read_confusion_matrix",
    "load_saved_ds_fit",
    "RESIDUAL_SIGMA_LIMIT",
    "fit_survival",
    "rb_decay_curve",
    "validate_fit_parameters",
    "validate_residuals",
    "validate_leakage_run",
]
