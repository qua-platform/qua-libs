"""Parameters for the CMA-ES gate optimisation node (03a_cmaes_gate_optimization).

The six-parameter search space (per qubit pair) is:

    amplitude_scale_target   —  multiplicative factor on the calibrated x90
                                amplitude for qubit_target (1.0 = unchanged).
    duration_offset_target   —  additive offset in ns for qubit_target.
    freq_detuning_target     —  additive frequency offset in Hz for qubit_target.

    amplitude_scale_control  —  multiplicative factor on the calibrated x90
                                amplitude for qubit_control (1.0 = unchanged).
    duration_offset_control  —  additive offset in ns for qubit_control.
    freq_detuning_control    —  additive frequency offset in Hz for qubit_control.

Each qubit in the pair has independent amplitude, duration, and frequency
parameters.  The same bounds (amplitude_scale_min/max, duration_offset_min/max,
freq_detuning_min/max) are applied to both qubits.

The optimisation node normalises all parameters to [0, 1] before running
CMA-ES, so sigma0 is in normalised units where 0.1 means 10 % of each
parameter's physical range.

The node loops over qubit_pairs: for each pair it runs RB on both qubits
(qubit_target and qubit_control) and optimises the average single-qubit
gate fidelity across the pair.  The fidelity is estimated from a 5-point
RB decay curve fitted with an exponential model P(m) = A·p^m + B using JAX.
"""

from __future__ import annotations

from typing import ClassVar

from qualibrate.core import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)

from .parameters import CMAESParameters


class CMAESGateOptParameters(CMAESParameters):
    """Parameters specific to the CMA-ES single-qubit gate optimisation."""

    num_shots: int = 100
    """Number of shots per circuit per depth per candidate. Default is 100.
    Together with num_circuits determines score precision: σ ≈ 1/√(shots×circuits).
    For high-fidelity (>99%) targets, consider 200+ shots."""
    num_circuits: int = 10
    """Number of random Clifford circuits per depth. Default is 10."""

    # --- RB depth configuration ---
    num_rb_depths: int = 7
    """Number of RB circuit depths to measure. Default is 7."""
    depth_min: int = 1
    """Shallowest RB depth (total Cliffords). Default is 1."""
    depth_max: int = 200
    """Deepest RB depth (total Cliffords). Default is 200.
    Depths are logarithmically spaced between depth_min and depth_max,
    placing more points where the exponential decay curve has the most
    curvature.  For high-fidelity gates (F>99%, p≈0.98) the default
    gives p^200≈0.018 — good dynamic range.  Reduce if expecting
    lower fidelity (F<95%)."""

    # --- Amplitude bounds (relative to current calibrated x90 amplitude) ---
    amplitude_scale_initial: float = 1.0
    """Initial amplitude scale factor (1.0 = current calibrated). Default is 1.0."""
    amplitude_scale_min: float = 0.8
    """Lower bound for amplitude scale. Default is 0.8."""
    amplitude_scale_max: float = 1.2
    """Upper bound for amplitude scale. Default is 1.2."""

    # --- Duration bounds (ns, must be multiple of 4) ---
    duration_offset_initial: float = 0.0
    """Initial duration offset from current calibrated length (ns). Default is 0."""
    duration_offset_min: float = -200.0
    """Lower bound for duration offset (ns). Default is -200.
    For ~1000 ns pulses this gives ±20% range, matching amplitude_scale bounds."""
    duration_offset_max: float = 200.0
    """Upper bound for duration offset (ns). Default is 200.
    For ~1000 ns pulses this gives ±20% range, matching amplitude_scale bounds."""

    # --- Frequency detuning bounds (Hz) ---
    freq_detuning_initial: float = 0.0
    """Initial frequency detuning from calibrated IF (Hz). Default is 0."""
    freq_detuning_min: float = -100e3
    """Lower bound for frequency detuning (Hz). Default is -100 kHz."""
    freq_detuning_max: float = 100e3
    """Upper bound for frequency detuning (Hz). Default is +100 kHz."""

    seed: int = 42
    """Seed for random Clifford circuit generation. Default is 42."""

    compilation_timeout: float = 600.0
    """Timeout in seconds for the QMM gRPC API calls. Default is 600 s."""

    # 6D search with noisy RB objective: population_size=14 gives robust
    # covariance estimation.  Rule of thumb: 4 + floor(3·ln(n)) = 9 for
    # n=6, but noisy objectives benefit from ~2× that.
    population_size: int = 14

    # All parameters normalised to [0, 1] before running, so sigma0 is in
    # normalised units (0.2 = 20% of each parameter's physical range).
    # This ensures CMA-ES explores all 6 dimensions adequately from the
    # first generation without being so wide that it wastes evaluations.
    sigma0: float = 0.2

    # 6D noisy objective with a flat fidelity landscape near the optimum
    # benefits from a generous generation budget.
    max_generations: int = 200

    # Override the generic CMAESParameters default: for gate fidelity, 0.5 is
    # the depolarising floor — any operating gate will pass.  99 % is a
    # meaningful minimum for single-qubit gate characterisation.
    success_threshold: float = 0.99


class CMAESGateParameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESGateOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 03a_cmaes_gate_optimization."""

    # This node operates on qubit pairs, so graph target injection must write
    # to `qubit_pairs` (the NodeParameters default is "qubits", which has no
    # matching field here and would break any graph that targets this node).
    targets_name: ClassVar[str] = "qubit_pairs"