"""RL-based single-qubit gate optimisation using 3-point randomized benchmarking.

Overview
--------
This node optimises the x90 pulse amplitude and duration by maximising
the Clifford gate fidelity.  The fidelity is estimated cheaply via the
**Analytical Decay Estimation (ADE)** method (Marciniak et al.,
arXiv:2602.11912), which extracts the RB decay parameter from only
three circuit depths — no curve fitting required.

Algorithm
---------
The optimiser maintains a multivariate Gaussian distribution
``N(μ, diag(σ²))`` over two parameters:

    θ = [amplitude_scale, duration_offset]

where ``amplitude_scale`` is a multiplicative factor applied to the
calibrated x90 (and x180) pulse amplitude, and ``duration_offset`` is
an additive offset in nanoseconds relative to the current pulse length.

At each iteration the algorithm:

1. **Evaluates** the score (ADE fidelity) at 5 parameter points:
   the current mean μ, plus ±ε perturbations along each dimension.
   ε = σ × epsilon_fraction.

2. **Estimates the gradient** via central finite differences::

       ∂F/∂θ_d ≈ (F(μ + ε·e_d) − F(μ − ε·e_d)) / (2·ε_d)

3. **Updates** the mean by gradient ascent (maximising fidelity)::

       μ ← μ + learning_rate · ∇F

   clipped to the configured parameter bounds.

4. **Shrinks** the exploration width::

       σ ← σ × sigma_decay

The loop terminates after ``max_iterations`` or when σ is negligibly
small.

Scoring — 3-point ADE
----------------------
Standard Clifford RB measures the survival probability P(m) at many
circuit depths m and fits P(m) = A·p^m + B.  ADE replaces this with a
closed-form estimate from exactly three depths:

    m₀,  m₀ + Δm,  m₀ + 3·Δm

The decay parameter p is computed as::

    c = (P(m₀ + 3Δm) − P(m₀)) / (P(m₀ + Δm) − P(m₀))
    p = (√(c − 3/4) − 1/2)^(1/Δm)
    F_clifford = (1 + p) / 2

This is SPAM-independent (the offset B and contrast A cancel out).

QUA program architecture
------------------------
A single QUA program is compiled per qubit and kept alive via
``infinite_loop_``.  Each iteration pushes new ``amplitude_scale`` and
``duration`` values through input streams; the OPX blocks on
``advance_input_stream`` until Python provides the next candidate.
No recompilation is needed across iterations.

For each candidate, the program runs the full 3-depth RB protocol:

    for depth in [m₀, m₀+Δm, m₀+3Δm]:
        for circuit in range(num_circuits):
            for shot in range(num_shots):
                empty → initialize → play Clifford sequence → measure

The Clifford sequences (random gates + recovery gate) are
**pre-generated in Python** and baked into QUA arrays.  Each native
gate is played via ``_play_rb_gate_scaled`` which passes the streamed
``amplitude_scale`` and ``duration`` to every XY gate call — virtual-Z
gates are unaffected.

Stream processing averages over shots, then circuits, producing one
survival probability per depth per candidate.

Parameter coupling
------------------
All single-qubit XY gates (x90, x180, x_neg90, y90, y180, y_neg90)
share a common amplitude/duration calibration via the ``XYDriveMacro``
system.  The y-axis gates are derived from x-axis gates with a
virtual-Z frame shift, and x180 amplitude = 2 × x90 amplitude.
Applying a uniform ``amplitude_scale`` factor preserves these ratios.

When the optimisation succeeds, ``update_state`` calls
``XYDriveMacro.update()`` which jointly updates both x90 and x180
pulse amplitudes and durations, keeping all derived gates consistent.
"""

# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam

from calibration_utils.cmaes import (
    OptimizationResult,
    plot_parameter_evolution,
    plot_score_convergence_on_ax,
)
from calibration_utils.cmaes.rl_gate_parameters import RLGateParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_qubits
from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    NUM_CLIFFORDS,
    NATIVE_GATE_MAP,
    compose_sequence,
    invert,
    decomposition,
)
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
        RL GATE OPTIMISATION — 3-POINT RB
Uses reinforcement-learning-style gradient descent to optimise single-qubit
gate parameters (x90 amplitude scale and duration offset) by maximising
the Clifford gate fidelity estimated from a 3-point randomised
benchmarking curve.

The 3-point RB uses the Analytical Decay Estimation (ADE) method from
Marciniak et al. (arXiv:2602.11912): survival probability is measured at
three circuit depths m₀, m₀+Δm, m₀+3Δm and the Clifford decay rate p is
extracted from a closed-form expression without curve fitting.

The optimisation follows a finite-difference policy gradient approach:
  1. Maintain a multivariate Gaussian N(μ, diag(σ²)) over parameters.
  2. At each iteration, evaluate the score at μ and at ±ε perturbations
     in each parameter dimension (central finite differences).
  3. Compute the gradient of the score w.r.t. μ.
  4. Update μ via gradient ascent:  μ ← μ + lr · ∇score.
  5. Shrink σ by a decay factor each iteration.

The QUA program is compiled once per qubit and uses input streams to push
new candidate parameters per iteration (no recompilation).

Prerequisites:
    - Calibrated x90 and x180 pulse parameters (amplitude, duration).
    - Calibrated initialization, measurement, and PSB threshold.
    - Native gate operations registered on qubit.xy channel.

State update:
    - Updates the x90/x180 pulse amplitude and duration via the
      XYDriveMacro.update() mechanism.
"""

node = QualibrationNode[RLGateParameters, Quam](
    name="03_rl_gate_optimization",
    description=description,
    parameters=RLGateParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[RLGateParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_iterations = 15
    node.parameters.qubits = ["q1"]
    node.parameters.num_shots = 50
    node.parameters.num_circuits = 5
    # node.parameters.simulate = True


node.machine = Quam.load()


# ── Helpers ──────────────────────────────────────────────────────────────

_PARAM_NAMES = ["amplitude_scale", "duration_offset"]


def _quantize_duration(val: float, min_val: int = 16) -> int:
    """Quantize a continuous duration to a multiple of 4 ns, clamped.

    The OPX requires pulse durations to be multiples of 4 ns.  This
    function rounds to the nearest valid duration and enforces a
    minimum of ``min_val`` ns (default 16 ns = 4 clock cycles) to
    stay within hardware limits.
    """
    return max(min_val, int(round(val / 4.0)) * 4)


def _generate_rb_circuits(
    num_circuits: int,
    depths: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Pre-generate all RB gate sequences for the given depths.

    For each (depth, circuit) pair, generates a standard RB sequence:

        1. Draw ``depth - 1`` random Cliffords (integers 0..23).
        2. Compose them to find the net Clifford.
        3. Append the inverse Clifford as the recovery gate.
        4. Decompose every Clifford into native gates (x90, x180, ...,
           z90, z180, z_neg90) using the pre-computed decomposition
           table from ``clifford_tables``.

    All sequences are concatenated into a single flat list so they can
    be loaded into a QUA ``declare(int, value=...)`` array.  Companion
    ``offsets`` and ``lengths`` arrays allow the QUA program to index
    into the flat list for any (depth, circuit) pair.

    Parameters
    ----------
    num_circuits : int
        Number of independent random circuits per depth.
    depths : list[int]
        Circuit depths (total number of Cliffords including recovery).
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    gates_flat : list[int]
        Concatenated native-gate integers for all (depth, circuit) pairs.
    offsets : list[int]
        Start index into gates_flat for each (depth_idx, circuit_idx).
        Layout: ``offsets[depth_idx * num_circuits + circuit_idx]``.
    lengths : list[int]
        Number of native gates for each sequence.
    """
    gates_flat = []
    offsets = []
    lengths = []

    for depth in depths:
        n_random = depth - 1
        for _ in range(num_circuits):
            random_cliffords = rng.integers(0, NUM_CLIFFORDS, size=n_random).tolist()
            net = compose_sequence(random_cliffords) if n_random > 0 else 0
            recovery = invert(net)
            full_sequence = list(random_cliffords) + [recovery]

            native_gates = []
            for cliff_idx in full_sequence:
                decomp = decomposition(cliff_idx)
                native_gates.extend(NATIVE_GATE_MAP[g] for g in decomp)

            offsets.append(len(gates_flat))
            lengths.append(len(native_gates))
            gates_flat.extend(native_gates)

    return gates_flat, offsets, lengths


def _play_rb_gate_scaled(qubit, gate_int, amplitude_scale=None, duration=None):
    """Play a single native gate with optional amplitude/duration override.

    This is a modified version of ``play_rb_gate`` from the standard RB
    utilities.  The key difference is that ``amplitude_scale`` and
    ``duration`` are forwarded to every XY-plane gate call, allowing
    the optimiser to evaluate arbitrary pulse parameters without
    recompilation.

    Virtual-Z gates (z90, z180, z_neg90) are frame-only operations with
    zero duration and are unaffected by the amplitude/duration override.

    Parameters
    ----------
    qubit : LDQubit
        Qubit object with registered gate macros.
    gate_int : QUA int variable
        Native gate index (0–8) matching the encoding in
        ``clifford_tables.NATIVE_GATE_MAP``.
    amplitude_scale : QUA fixed variable or None
        Multiplicative factor applied to the pulse amplitude.
        A value of 1.0 plays the pulse at its calibrated amplitude.
        Because x180 is defined as 2× the x90 amplitude, applying the
        same scale factor preserves the correct ratio.
    duration : QUA int variable or None
        Absolute pulse duration in nanoseconds (must be a multiple of
        4 ns).  Overrides the pulse's configured length.
    """
    with switch_(gate_int, unsafe=True):
        with case_(0):
            qubit.x90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(1):
            qubit.x180(amplitude_scale=amplitude_scale, duration=duration)
        with case_(2):
            qubit.x_neg90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(3):
            qubit.y90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(4):
            qubit.y180(amplitude_scale=amplitude_scale, duration=duration)
        with case_(5):
            qubit.y_neg90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(6):
            qubit.z90()
        with case_(7):
            qubit.z180()
        with case_(8):
            qubit.z_neg90()


def _compute_ade_fidelity(
    p_m0: float, p_m1: float, p_m2: float, delta_m: int
) -> float:
    """Compute Clifford gate fidelity from 3-point ADE.

    The standard RB model is ``P(m) = A·p^m + B`` where ``p`` is the
    depolarising parameter.  Given three carefully spaced depths the
    SPAM parameters A and B cancel out, yielding a closed-form
    expression for ``p`` (Marciniak et al., arXiv:2602.11912):

    1. Compute the ratio ``c``::

           c = (P(m₀ + 3Δm) − P(m₀)) / (P(m₀ + Δm) − P(m₀))

       This is a ratio of differences, eliminating B; the contrast A
       cancels because it factors out identically from numerator and
       denominator (since the depths are chosen to ensure this).

    2. Extract the per-step decay::

           p = (√(c − 3/4) − 1/2)^{1/Δm}

    3. Convert to average Clifford gate fidelity::

           F_clifford = (1 + p) / 2

    Parameters
    ----------
    p_m0, p_m1, p_m2 : float
        Measured survival probabilities at depths m₀, m₀+Δm, m₀+3Δm.
    delta_m : int
        The depth spacing Δm.

    Returns
    -------
    float
        Clifford gate fidelity, or ``NaN`` if the computation is
        invalid (e.g. zero denominator or negative sqrt argument,
        which can happen with noisy data).
    """
    denom = p_m1 - p_m0
    if abs(denom) < 1e-12:
        return np.nan
    c = (p_m2 - p_m0) / denom
    arg = c - 0.75
    if arg < 0:
        return np.nan
    inner = np.sqrt(arg) - 0.5
    if inner <= 0:
        return np.nan
    p = inner ** (1.0 / delta_m)
    return (1.0 + p) / 2.0


# %% {Create_QUA_program}
def _build_qua_program(node, qubit, gates_flat, offsets, lengths, depths):
    """Build a QUA program for 3-point RB with streamed pulse parameters.

    The program uses ``infinite_loop_`` so it stays alive across
    optimisation iterations.  ``advance_input_stream`` blocks until
    Python pushes new candidate parameters, so the OPX idles between
    iterations with zero recompilation overhead.  Python cancels the
    job when optimisation is complete.

    Program structure (per candidate)::

        advance_input_stream(amplitude_scale, duration)
        for depth in [m₀, m₀+Δm, m₀+3Δm]:
            for circuit in range(num_circuits):
                for shot in range(num_shots):
                    reset_frame → align
                    empty → initialize → align
                    for gate in sequence[depth][circuit]:
                        play_rb_gate_scaled(gate, amp, dur)
                    align → measure → compensation → save

    Stream processing chain::

        raw shots → buffer(num_shots) → average
                  → buffer(num_circuits) → average
                  → buffer(n_depths) → save_all("survival")

    This produces one array of shape ``(n_depths,)`` per candidate,
    containing the survival probability at each of the 3 RB depths.

    Parameters
    ----------
    node : QualibrationNode
        Provides ``num_circuits`` and ``num_shots`` from parameters.
    qubit : LDQubit
        The qubit to benchmark.
    gates_flat : list[int]
        Flat array of pre-generated native-gate integers.
    offsets : list[int]
        Start offsets into ``gates_flat`` for each (depth, circuit).
    lengths : list[int]
        Number of gates per sequence.
    depths : list[int]
        The 3 RB circuit depths [m₀, m₀+Δm, m₀+3Δm].
    """
    num_circuits = node.parameters.num_circuits
    num_shots = node.parameters.num_shots
    n_depths = len(depths)

    with program() as qua_prog:
        amp_in = declare_input_stream("client", stream_id="amplitude_scale", dtype=fixed)
        dur_in = declare_input_stream("client", stream_id="duration", dtype=int)

        amp_scale = declare(fixed)
        dur = declare(int)

        gates_qua = declare(int, value=gates_flat)
        offsets_qua = declare(int, value=offsets)
        lengths_qua = declare(int, value=lengths)

        depth_idx = declare(int)
        circuit_idx = declare(int)
        shot_idx = declare(int)
        gate_idx = declare(int)
        current_gate = declare(int)
        seq_index = declare(int)
        seq_offset = declare(int)
        seq_length = declare(int)

        state = declare(int)
        state_st = declare_output_stream()

        with infinite_loop_():
            advance_input_stream(amp_in)
            advance_input_stream(dur_in)
            assign(amp_scale, amp_in)
            assign(dur, dur_in)

            with for_(depth_idx, 0, depth_idx < n_depths, depth_idx + 1):
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    assign(seq_index, depth_idx * num_circuits + circuit_idx)
                    assign(seq_offset, offsets_qua[seq_index])
                    assign(seq_length, lengths_qua[seq_index])

                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit.xy.name)
                        align()
                        qubit.empty()
                        qubit.initialize(
                            conditional_drive=True,
                        )
                        align()

                        with for_(gate_idx, 0, gate_idx < seq_length, gate_idx + 1):
                            assign(current_gate, gates_qua[seq_offset + gate_idx])
                            _play_rb_gate_scaled(qubit, current_gate, amp_scale, dur)

                        align()
                        p = qubit.measure()
                        align()
                        qubit.voltage_sequence.apply_compensation_pulse(
                            go_to_zero=True, return_to_zero=True
                        )
                        align()

                        assign(state, Cast.to_int(p))
                        save(state, state_st)

        with stream_processing():
            (
                state_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(n_depths)
                .save_all("survival")
            )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[RLGateParameters, Quam]):
    """Validate parameters, compute the 3 RB depths, and compile the QUA program.

    This action resolves the target qubits, computes the three ADE
    depths ``[m₀, m₀+Δm, m₀+3Δm]``, pre-generates the Clifford gate
    sequences, and builds the QUA program.  All artefacts are stored
    in ``node.namespace`` for use by subsequent actions.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)
    if not qubits:
        raise ValueError("No qubits resolved — check qubits parameter or machine config.")

    m0 = node.parameters.depth_start
    dm = node.parameters.depth_delta
    depths = [m0, m0 + dm, m0 + 3 * dm]
    node.namespace["depths"] = depths

    rng = np.random.default_rng(node.parameters.seed)

    qubit = qubits[0]
    gates_flat, offsets, lengths = _generate_rb_circuits(
        node.parameters.num_circuits, depths, rng,
    )
    node.namespace["rb_circuits"] = {
        "gates_flat": gates_flat,
        "offsets": offsets,
        "lengths": lengths,
    }

    node.namespace["qua_program"] = _build_qua_program(
        node, qubit, gates_flat, offsets, lengths, depths,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[RLGateParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Run_RL_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_rl_loop(node: QualibrationNode[RLGateParameters, Quam]):
    """Execute finite-difference gradient descent optimisation per qubit.

    For each qubit the algorithm proceeds as follows:

    **Setup**
      - Pre-generate Clifford RB circuits for the 3 target depths.
      - Compile and start the QUA program (runs in ``infinite_loop_``).
      - Initialise μ (mean) and σ (std) of the search distribution
        from the node parameters.

    **Iteration loop** (up to ``max_iterations``):

      Both μ and σ are maintained in a [0, 1]-normalised parameter space
      (amplitude_scale and duration_offset divided by their respective
      ranges).  This ensures the learning rate and sigma_decay apply
      uniformly to both dimensions regardless of their physical scale.
      Candidates are denormalised to physical units before being sent to
      the OPX; history and logs are stored in physical units.

      1. Compute the perturbation step  ``ε = σ_norm × epsilon_fraction``.
      2. Build 5 candidate points in normalised space:
         ``[μ, μ+ε₀·ê₀, μ−ε₀·ê₀, μ+ε₁·ê₁, μ−ε₁·ê₁]``
         Each candidate is clipped to [0, 1], then denormalised.
      3. Push each denormalised candidate to the QUA program via input
         streams and wait for all 3-depth RB results.
      4. Convert survival probabilities → ADE fidelity scores.
      5. Compute the gradient in normalised space via central finite
         differences::

             ∂F/∂θ̂_d = (F(μ+ε·ê_d) − F(μ−ε·ê_d)) / (2·ε_d)

      6. Gradient ascent update in normalised space: ``μ ← μ + lr · ∇F``.
      7. Shrink exploration: ``σ_norm ← σ_norm × sigma_decay``.
      8. Track the best score and parameters across all iterations.

    **Outputs** (stored in ``node.namespace`` and ``node.results``):
      - ``optimization_results``: per-qubit ``OptimizationResult``
        containing full history of parameters, scores, and candidates.
      - ``measurement_streams``: per-qubit dict of survival
        probabilities at each depth for every iteration, enabling
        the survival-stream plot.
    """
    import time as _time

    qubits = node.namespace["qubits"]
    depths = node.namespace["depths"]
    dm = node.parameters.depth_delta

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    n_params = len(_PARAM_NAMES)
    max_iter = node.parameters.max_iterations
    lr = node.parameters.learning_rate
    eps_frac = node.parameters.epsilon_fraction
    sigma_decay = node.parameters.sigma_decay

    optimization_results = {}
    measurement_streams = {}

    for qubit in qubits:
        node.log(f"  Starting RL gate optimisation for {qubit.name}...")

        rng = np.random.default_rng(node.parameters.seed)
        gates_flat, offsets, lengths = _generate_rb_circuits(
            node.parameters.num_circuits, depths, rng,
        )

        qua_prog = _build_qua_program(
            node, qubit, gates_flat, offsets, lengths, depths,
        )

        calibrated_duration = qubit.macros["x90"].pulse.length

        # Normalise both parameters to [0, 1] so that the learning rate and
        # sigma values are meaningful for both dimensions.  Without this,
        # amplitude updates (~0.025 per step) vastly dominate duration updates
        # (~0.00025 ns per step, below the 4 ns quantisation limit).
        lo = np.array([
            node.parameters.amplitude_scale_min,
            node.parameters.duration_offset_min,
        ])
        hi = np.array([
            node.parameters.amplitude_scale_max,
            node.parameters.duration_offset_max,
        ])
        param_range = hi - lo

        mu_phys = np.array([
            node.parameters.amplitude_scale_initial,
            node.parameters.duration_offset_initial,
        ])
        mu_norm = (mu_phys - lo) / param_range

        sigma_norm = np.array([
            node.parameters.amplitude_sigma,
            node.parameters.duration_sigma,
        ]) / param_range

        param_history = []
        score_history = []
        all_candidates = []
        all_scores = []
        best_score_so_far = -np.inf
        best_params_so_far = mu_phys.copy()

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            result_handles = job.result_handles
            survival_handle = result_handles.get("survival")
            gen_counter = 0

            survival_streams = {"depth_0": [], "depth_1": [], "depth_2": []}

            def _evaluate_candidates(candidates: np.ndarray) -> np.ndarray:
                """Push candidates to the QUA program and return ADE fidelity scores.

                For each candidate ``[amplitude_scale, duration_offset]``:

                1. Push ``amplitude_scale`` (fixed-point) and the quantised
                   absolute duration (int) to the OPX input streams.
                2. Wait until all results arrive via the ``survival``
                   output stream.
                3. Fetch the 3-element survival array ``[P(m₀), P(m₁), P(m₂)]``
                   and compute the ADE fidelity.

                Also records the per-depth survival probabilities in
                ``survival_streams`` for later plotting.
                """
                nonlocal gen_counter
                n_cand = len(candidates)
                for c in candidates:
                    job.push_to_input_stream("amplitude_scale", float(c[0]))
                    dur_abs = calibrated_duration + c[1]
                    job.push_to_input_stream(
                        "duration", _quantize_duration(dur_abs)
                    )

                target_count = gen_counter + n_cand
                while survival_handle.count_so_far() < target_count:
                    _time.sleep(0.005)

                scores = np.empty(n_cand)
                survs_d0 = np.empty(n_cand)
                survs_d1 = np.empty(n_cand)
                survs_d2 = np.empty(n_cand)
                for i in range(n_cand):
                    surv = np.asarray(
                        survival_handle.fetch(gen_counter, flat_struct=True),
                        dtype=np.float64,
                    )
                    gen_counter += 1
                    survs_d0[i], survs_d1[i], survs_d2[i] = surv[0], surv[1], surv[2]
                    scores[i] = _compute_ade_fidelity(surv[0], surv[1], surv[2], dm)

                survival_streams["depth_0"].append(survs_d0.copy())
                survival_streams["depth_1"].append(survs_d1.copy())
                survival_streams["depth_2"].append(survs_d2.copy())

                return scores

            try:
                for iteration in range(max_iter):
                    # Work entirely in normalised [0, 1] space so that lr and
                    # sigma_decay apply uniformly across amplitude and duration.
                    eps_norm = sigma_norm * eps_frac

                    candidates_norm = [mu_norm.copy()]
                    for d in range(n_params):
                        e_d = np.zeros(n_params)
                        e_d[d] = eps_norm[d]
                        candidates_norm.append(np.clip(mu_norm + e_d, 0.0, 1.0))
                        candidates_norm.append(np.clip(mu_norm - e_d, 0.0, 1.0))
                    candidates_norm = np.array(candidates_norm)

                    # Denormalise before sending to the OPX.
                    candidates_phys = lo + candidates_norm * param_range
                    scores = _evaluate_candidates(candidates_phys)

                    non_finite = ~np.isfinite(scores)
                    if non_finite.any():
                        scores = np.where(non_finite, 0.0, scores)

                    center_score = scores[0]

                    # Gradient in normalised space — both components are now
                    # dimensionless so lr applies equally to amplitude and duration.
                    grad_norm = np.zeros(n_params)
                    for d in range(n_params):
                        f_plus = scores[1 + 2 * d]
                        f_minus = scores[2 + 2 * d]
                        denom = 2 * eps_norm[d]
                        if abs(denom) > 1e-12:
                            grad_norm[d] = (f_plus - f_minus) / denom

                    mu_norm = mu_norm + lr * grad_norm
                    mu_norm = np.clip(mu_norm, 0.0, 1.0)

                    sigma_norm *= sigma_decay

                    # Denormalise for history and logging.
                    mu_phys = lo + mu_norm * param_range

                    gen_best_idx = int(np.argmax(scores))
                    gen_best_score = float(scores[gen_best_idx])
                    if gen_best_score > best_score_so_far:
                        best_score_so_far = gen_best_score
                        best_params_so_far = candidates_phys[gen_best_idx].copy()

                    param_history.append(mu_phys.copy())
                    score_history.append(best_score_so_far)
                    all_candidates.append(candidates_phys.copy())
                    all_scores.append(scores.copy())

                    pct = 100.0 * (iteration + 1) / max_iter
                    param_str = ", ".join(
                        f"{name}={val:.6g}"
                        for name, val in zip(_PARAM_NAMES, mu_phys)
                    )
                    node.log(
                        f"  [{qubit.name}] iter {iteration+1}/{max_iter} "
                        f"({pct:5.1f}%) | best = {best_score_so_far:.6f} | "
                        f"center = {center_score:.6f} | "
                        f"|∇| = {np.linalg.norm(grad_norm):.4g} | μ: {param_str}"
                    )

                    if np.all(sigma_norm < 1e-8):
                        node.log(f"  [{qubit.name}] σ converged — stopping.")
                        break

            finally:
                job.cancel()

        n_gen = len(score_history)
        opt_result = OptimizationResult(
            best_params=best_params_so_far,
            best_score=best_score_so_far,
            param_names=_PARAM_NAMES,
            param_history=param_history,
            score_history=score_history,
            all_candidates=all_candidates,
            all_scores=all_scores,
            n_generations=n_gen,
            converged=bool(np.all(sigma_norm < 1e-8)),
            stop_reason=(
                "sigma_converged" if bool(np.all(sigma_norm < 1e-8))
                else f"maxiter: {max_iter}"
            ),
        )
        optimization_results[qubit.name] = opt_result
        measurement_streams[qubit.name] = survival_streams

    node.namespace["optimization_results"] = optimization_results
    node.namespace["measurement_streams"] = measurement_streams
    node.results["optimization_results"] = {
        name: result.to_dict() for name, result in optimization_results.items()
    }
    node.results["measurement_streams"] = {
        name: {k: [arr.tolist() for arr in v] for k, v in streams.items()}
        for name, streams in measurement_streams.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[RLGateParameters, Quam]):
    """Load a previously saved optimisation result.

    Restores ``optimization_results`` and ``measurement_streams``
    from ``node.results``, reconstructing numpy arrays from the
    serialised lists so that ``analyse_data`` and ``plot_data`` work
    identically to a live run.
    """
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["optimization_results"] = {
        name: OptimizationResult.from_dict(d)
        for name, d in node.results["optimization_results"].items()
    }
    raw_streams = node.results.get("measurement_streams", {})
    node.namespace["measurement_streams"] = {
        name: {k: [np.array(arr) for arr in v] for k, v in streams.items()}
        for name, streams in raw_streams.items()
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RLGateParameters, Quam]):
    """Summarise the RL optimisation outcome per qubit.

    Delegates to ``analyse_optimization`` (shared with CMA-ES nodes)
    which checks whether the best score exceeds ``success_threshold``.
    Sets ``node.outcomes`` to ``SUCCESSFUL`` or ``FAILED`` per qubit
    and logs the best parameters and fidelity.
    """
    from calibration_utils.cmaes import analyse_optimization, log_optimization_results

    opt_results = node.namespace["optimization_results"]

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )
    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    node.outcomes = {
        qname: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qname, summary in fit_results.items()
    }


# %% {Plot_data}
def _plot_rb_survival_streams_on_ax(
    ax: plt.Axes,
    streams: dict,
    qubit_name: str = "",
) -> None:
    """Plot survival probability at each RB depth (mean ± std) vs iteration.

    At each iteration the optimiser evaluates multiple candidates (the
    centre point and ±ε perturbations).  This function shows, for each
    of the 3 ADE depths, how the average survival probability evolves
    as the optimiser tunes the gate parameters.

    Shaded bands indicate the standard deviation across candidates
    within each iteration — a shrinking band signals convergence.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    streams : dict
        Keys ``"depth_0"``, ``"depth_1"``, ``"depth_2"``, each mapping
        to a list of arrays (one array of survival probabilities per
        iteration, with one entry per candidate).
    qubit_name : str, optional
        Appended to the subplot title for identification.
    """
    labels_colors = [
        ("depth_0", "P(m₀)", "C0"),
        ("depth_1", "P(m₀+Δm)", "C1"),
        ("depth_2", "P(m₀+3Δm)", "C2"),
    ]
    for key, label, color in labels_colors:
        data = streams[key]
        n_gen = len(data)
        generations = np.arange(1, n_gen + 1)
        means = np.array([np.mean(arr) for arr in data])
        stds = np.array([np.std(arr) for arr in data])

        ax.fill_between(
            generations, means - stds, means + stds,
            alpha=0.2, color=color,
        )
        ax.plot(generations, means, "o-", color=color, markersize=4, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Survival probability")
    title = (
        f"RB survival streams — {qubit_name}" if qubit_name
        else "RB survival streams"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RLGateParameters, Quam]):
    """Generate convergence and parameter-evolution plots (per qubit).

    Two figures are produced:

    **fig_combined** (2 × n_qubits subplots):
      - Top row: RB survival streams — mean ± std of the survival
        probability at each of the 3 ADE depths across candidates,
        plotted as a function of iteration.  Tracks whether the raw
        RB probabilities improve over time.
      - Bottom row: ADE fidelity convergence — mean ± std of the
        score (across candidates per iteration) plus the best-so-far
        trace.  This is the primary convergence diagnostic.

    **fig_params** (from ``plot_parameter_evolution``):
      - Shows how μ (mean parameters) evolve over iterations, with
        the spread of candidates in each generation.
    """
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    qubit_names = list(opt_results.keys())
    n_qubits = max(len(qubit_names), 1)

    fig_combined, axes = plt.subplots(
        2, n_qubits, figsize=(7 * n_qubits, 8), squeeze=False,
    )

    for col, qname in enumerate(qubit_names):
        streams = measurement_streams.get(qname)
        if streams:
            _plot_rb_survival_streams_on_ax(axes[0, col], streams, qname)
        else:
            axes[0, col].set_title(f"No survival data — {qname}")

        plot_score_convergence_on_ax(axes[1, col], opt_results[qname], qname)

    fig_combined.tight_layout()

    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    node.results["figures"] = {
        "fidelity_and_convergence": fig_combined,
        "parameter_evolution": fig_params,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[RLGateParameters, Quam]):
    """Apply optimal x90 amplitude and duration via XYDriveMacro.update().

    For each qubit that converged successfully:

    1. Retrieves the best ``amplitude_scale`` and ``duration_offset``
       from the fit results.
    2. Quantises the new absolute duration (current + offset) to a
       4 ns multiple.
    3. Calls ``XYDriveMacro.update(amplitude_scale, duration)`` on the
       x90 macro, which internally:
       - Multiplies the x90 pulse amplitude by ``amplitude_scale``.
       - Sets x180 amplitude = 2 × new x90 amplitude.
       - Sets the pulse duration for both x90 and x180.
       This automatically propagates to all derived gates (y90,
       y_neg90, y180, etc.) because they share the same underlying
       amplitude/duration via the macro system.

    Qubits whose optimisation did not meet the ``success_threshold``
    are skipped with a log warning.
    """
    fit_results = node.results["fit_results"]

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            pair_summary = fit_results.get(qubit.name)
            if pair_summary is None:
                continue
            if not pair_summary["success"]:
                node.log(
                    f"  {qubit.name}: optimisation did not succeed — skipping."
                )
                continue

            best = pair_summary["best_params"]
            opt_amp_scale = best["amplitude_scale"]
            opt_dur_offset = best["duration_offset"]

            xy_macro = qubit.macros["x90"]

            current_x90_amp = xy_macro.pulse.amplitude
            current_pi_amp = xy_macro.pi_pulse.amplitude
            current_duration = xy_macro.pulse.length

            new_duration = _quantize_duration(current_duration + opt_dur_offset)

            xy_macro.update(
                amplitude_scale=opt_amp_scale,
                duration=new_duration,
            )

            node.log(
                f"  {qubit.name}: gate params updated — "
                f"x90_amp: {current_x90_amp:.6g} → {current_x90_amp * opt_amp_scale:.6g} V, "
                f"x180_amp: {current_pi_amp:.6g} → {current_pi_amp * opt_amp_scale:.6g} V, "
                f"duration: {current_duration} → {new_duration} ns, "
                f"fidelity={pair_summary['best_score']:.6f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RLGateParameters, Quam]):
    """Persist all results, figures, and parameters to disk."""
    node.save()
