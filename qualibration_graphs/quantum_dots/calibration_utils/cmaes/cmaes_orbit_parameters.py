"""Parameters for the CMA-ES orbit optimisation node (03b_cmaes_orbit).

The six-parameter search space (per qubit pair) is identical to the gate
optimisation node:

    amplitude_scale_target   —  multiplicative factor on the calibrated x90
                                amplitude for qubit_target (1.0 = unchanged).
    duration_offset_target   —  additive offset in ns for qubit_target.
    freq_detuning_target     —  additive frequency offset in Hz for qubit_target.

    amplitude_scale_control  —  multiplicative factor on the calibrated x90
                                amplitude for qubit_control (1.0 = unchanged).
    duration_offset_control  —  additive offset in ns for qubit_control.
    freq_detuning_control    —  additive frequency offset in Hz for qubit_control.

Scoring — orbit separation
--------------------------
Instead of fitting a multi-depth RB decay, the orbit cost function measures
survival probability at a single fixed Clifford depth under two preparations:

    P_normal:  initialize in |0⟩ → random Clifford sequence → measure
    P_pi:      initialize in |0⟩ → π-pulse → random Clifford sequence → measure

The two variants use independent random circuit instances.  The score is:

    score = P_normal − P_pi

This equals p^m (the RB depolarizing parameter raised to the chosen depth),
which is maximised when gate fidelity is highest.
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


class CMAESOrbitOptParameters(CMAESParameters):
    """Parameters specific to the CMA-ES orbit optimisation."""

    num_shots: int = 100
    """Number of shots per circuit per candidate. Default is 100."""
    num_circuits: int = 30
    """Number of random Clifford circuits per variant (normal / pi-pulse).
    More circuits reduce variance of the survival estimate. Default is 30."""

    orbit_depth: int = 40
    """Fixed Clifford depth for the orbit sequences. Default is 40.

    The optimal depth maximises the Fisher information of the orbit score
    with respect to gate fidelity F.  The orbit signal is p^m (where
    p = 2F − 1) and the shot-noise variance is (1 − p^(2m)) / (2·N_eff).
    Maximising sensitivity d(score)/dF / σ gives the condition:

        1 + m·ln(p) = p^(2m)

    For small error rate r = 1−p this simplifies to m* ≈ 0.8 / r, i.e.:

        m* = 0.8 / (2·(1 − F))

    At F = 99%  (p = 0.98, r = 0.02):  m* = 40   → score ≈ 0.45
    At F = 99.5% (p = 0.99, r = 0.01): m* = 80   → score ≈ 0.45
    At F = 98%  (p = 0.96, r = 0.04):  m* = 20   → score ≈ 0.44

    The default of 40 is optimal for discriminating around 99% fidelity."""

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
    """Lower bound for duration offset (ns). Default is -200."""
    duration_offset_max: float = 200.0
    """Upper bound for duration offset (ns). Default is 200."""

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

    population_size: int = 14
    sigma0: float = 0.2
    max_generations: int = 200

    success_threshold: float = 0.3
    """Minimum separation score to consider the optimisation successful.
    At depth 40: p^40 = 0.98^40 ≈ 0.45 for F=99%.  A threshold of 0.3
    corresponds to F ≈ 98.5% (p ≈ 0.970), providing a reasonable floor.
    Default is 0.3."""


class CMAESOrbitParameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESOrbitOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 03b_cmaes_orbit."""

    # This node operates on qubit pairs, so graph target injection must write
    # to `qubit_pairs` (the NodeParameters default is "qubits", which has no
    # matching field here and would break any graph that targets this node).
    targets_name: ClassVar[str] = "qubit_pairs"
