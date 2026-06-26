"""Parameters for the RL gate optimisation node (03_rl_gate_optimization).

This module defines two parameter classes:

``RLGateOptParameters``
    RL-specific hyperparameters: learning rate, sigma decay, finite-
    difference step, RB depths, amplitude/duration bounds, etc.
    Inherits from ``CMAESParameters`` to reuse ``success_threshold``
    and ``timeout`` which are shared with CMA-ES optimisation nodes.

``RLGateParameters``
    Composite parameter set that combines ``RLGateOptParameters`` with
    ``CommonNodeParameters`` (load_data_id, simulate, …) and
    ``QubitsExperimentNodeParameters`` (qubit selection).  This is the
    concrete type used by the node.

The two-parameter search space is:

    amplitude_scale  —  multiplicative factor on the calibrated x90
                        amplitude (1.0 = unchanged).  Bounded by
                        [amplitude_scale_min, amplitude_scale_max].

    duration_offset  —  additive offset in nanoseconds relative to the
                        calibrated pulse length.  The absolute duration
                        is quantised to 4 ns multiples by the QUA
                        program.  Bounded by [duration_offset_min,
                        duration_offset_max].
"""

from __future__ import annotations

from qualibrate.core import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
)

from .parameters import CMAESParameters


class RLGateOptParameters(CMAESParameters):
    """Parameters specific to the RL-based single-qubit gate optimisation.

    The optimiser tunes the x90 pulse amplitude and duration to maximise
    the Clifford gate fidelity estimated via 3-point analytical decay
    estimation (ADE) on a randomized benchmarking curve.

    The search distribution is a diagonal Gaussian N(μ, diag(σ²)) over
    two parameters.  The ``*_initial`` fields set μ₀, the ``*_sigma``
    fields set σ₀, and the ``*_min``/``*_max`` fields define hard
    bounds applied after every gradient update.
    """

    num_shots: int = 100
    """Number of shots per circuit per depth per candidate. Default is 100."""
    num_circuits: int = 10
    """Number of random Clifford circuits per depth. Default is 10."""

    # --- 3-point ADE depths: m0, m0+Δm, m0+3Δm ---
    depth_start: int = 1
    """First RB depth m₀ (total Cliffords). Default is 1."""
    depth_delta: int = 50
    """Spacing Δm between depths. The 3 depths are m₀, m₀+Δm, m₀+3Δm. Default is 50."""

    # --- Amplitude bounds (relative to current calibrated x90 amplitude) ---
    amplitude_scale_initial: float = 1.0
    """Initial amplitude scale factor (1.0 = current calibrated). Default is 1.0."""
    amplitude_scale_min: float = 0.5
    """Lower bound for amplitude scale. Default is 0.5."""
    amplitude_scale_max: float = 1.5
    """Upper bound for amplitude scale. Default is 1.5."""
    amplitude_sigma: float = 0.05
    """Initial Gaussian std for amplitude scale parameter. Default is 0.05."""

    # --- Duration bounds (ns, must be multiple of 4) ---
    duration_offset_initial: float = 0.0
    """Initial duration offset from current calibrated length (ns). Default is 0."""
    duration_offset_min: float = -40.0
    """Lower bound for duration offset (ns). Default is -40."""
    duration_offset_max: float = 40.0
    """Upper bound for duration offset (ns). Default is 40."""
    duration_sigma: float = 8.0
    """Initial Gaussian std for duration offset (ns). Default is 8.0."""

    # --- RL hyper-parameters ---
    learning_rate: float = 0.1
    """Gradient descent step size. Default is 0.1."""
    sigma_decay: float = 0.98
    """Multiplicative decay of σ per iteration. Default is 0.98."""
    epsilon_fraction: float = 0.5
    """Finite-difference step as fraction of current σ. Default is 0.5."""
    max_iterations: int = 30
    """Maximum number of gradient-descent iterations. Default is 30."""

    seed: int = 42
    """Seed for random Clifford circuit generation. Default is 42."""

    compilation_timeout: float = 600.0
    """Timeout in seconds for the QMM gRPC API calls. Default is 600 s."""

    # Override the generic CMAESParameters default: for gate fidelity, 0.5 is
    # the depolarising floor — any operating gate will pass.  99% is a
    # meaningful minimum for single-qubit gate characterisation.
    success_threshold: float = 0.99


class RLGateParameters(
    NodeParameters,
    CommonNodeParameters,
    RLGateOptParameters,
    QubitsExperimentNodeParameters,
):
    """Composite parameter set for 03_rl_gate_optimization.

    Inherits fields from all four bases via MRO:

    - ``NodeParameters``: core Qualibrate node interface.
    - ``CommonNodeParameters``: ``load_data_id``, ``simulate``, etc.
    - ``RLGateOptParameters``: RL hyperparameters + RB settings
      (which itself inherits ``CMAESParameters`` for
      ``success_threshold`` and ``timeout``).
    - ``QubitsExperimentNodeParameters``: ``qubits`` list selection.
    """
