"""CMA-ES single-qubit gate optimisation using 5-point randomized benchmarking.

Overview
--------
This node optimises the x90 pulse amplitude, duration, and drive frequency
independently for each qubit in a pair, maximising the average Clifford
gate fidelity.  For each pair, RB is run on both qubit_target and
qubit_control; the optimiser maximises the mean fidelity of the two qubits.

The fidelity is estimated by fitting a 5-point exponential decay curve
P(m) = A·p^m + B to the RB survival probabilities using JAX-based batched
nonlinear least squares.

Algorithm
---------
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) maintains a
multivariate Gaussian distribution over six parameters:

    θ = [amplitude_scale_target, duration_offset_target, freq_detuning_target,
         amplitude_scale_control, duration_offset_control, freq_detuning_control]

Each qubit has its own amplitude_scale (multiplicative factor on the
calibrated x90/x180 amplitude), duration_offset (additive offset in
nanoseconds relative to the calibrated pulse length), and freq_detuning
(additive frequency offset in Hz relative to the calibrated IF).

At each generation, CMA-ES:

1. **Samples** ``population_size`` candidate parameter vectors from its
   current distribution N(μ, σ²·C).
2. **Evaluates** all candidates in a single QUA program execution
   (no recompilation between generations).
3. **Updates** μ, σ, and C via the CMA-ES recombination and adaptation
   rules, adapting both the step size and the full covariance matrix.

The loop terminates after ``max_generations`` or when CMA-ES convergence
criteria (``tolx``, ``tolfun``) are satisfied.

Scoring — 5-point exponential decay fit (JAX-batched)
-----------------------------------------------------
Standard Clifford RB measures the survival probability P(m) at several
circuit depths m and fits P(m) = A·p^m + B.  This node measures at
``num_rb_depths`` logarithmically spaced depths between ``depth_min``
and ``depth_max``.  Logarithmic spacing concentrates points where the
exponential decay has the most curvature, giving a better-conditioned
fit — especially with few points.

The exponential model is fitted using JAX ``vmap`` over candidates and
qubits simultaneously for efficient batched evaluation.  The Clifford
gate fidelity is F = (1 + p) / 2.

For each candidate, the fidelity is computed per-qubit in the pair
and the score is the average:

    score = (F_target + F_control) / 2

QUA program architecture
------------------------
A single QUA program is compiled per qubit pair and kept alive via
``infinite_loop_``.  For each candidate the program receives independent
amplitude/duration parameters for each qubit and runs RB sequentially:
first qubit_target, then qubit_control.

For each qubit and candidate:

    for depth in depths:
        for circuit in range(num_circuits):
            for shot in range(num_shots):
                empty → initialize → play Clifford sequence → measure

Stream processing produces arrays of shape ``(pop_size, n_depths)``
per qubit per generation.

Prerequisites:
    - Calibrated x90 and x180 pulse parameters (amplitude, duration).
    - Calibrated initialization, measurement, and PSB threshold.
    - Native gate operations registered on qubit.xy channel.

State update:
    - Updates the x90/x180 pulse amplitude and duration via the
      XYDriveMacro.update() mechanism independently on each qubit.
    - The frequency detuning is optimised but not applied to the qubit
      state; the original intermediate frequency is preserved.
"""

# %% {Imports}
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam

from calibration_utils.cmaes import (
    OptimizationResult,
    run_cmaes_optimization,
    analyse_optimization,
    log_optimization_results,
    plot_parameter_evolution,
    plot_score_convergence_on_ax,
)
from calibration_utils.cmaes.cmaes_gate_parameters import CMAESGateParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_qubit_pairs
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
        CMA-ES GATE OPTIMISATION — 5-POINT RB (QUBIT-PAIR, PER-QUBIT PARAMS)
Uses CMA-ES to optimise single-qubit gate parameters (x90 amplitude scale,
duration offset, and frequency detuning) independently for each qubit in a
pair, maximising the average Clifford gate fidelity across both qubits.

The search space is 6-dimensional:
    [amp_scale_target, dur_offset_target, freq_detuning_target,
     amp_scale_control, dur_offset_control, freq_detuning_control]

RB is run on qubit_target and qubit_control for each candidate.  The
fidelity per qubit is estimated by fitting a 5-point exponential decay
P(m) = A·p^m + B using JAX-batched nonlinear least squares.  The score
is the mean fidelity of the pair: (F_target + F_control) / 2.

CMA-ES evaluates a full population of candidates per generation; all
candidates are pushed to the OPX in a single compiled program execution
via input streams (no recompilation between generations).

Prerequisites:
    - Calibrated x90 and x180 pulse parameters (amplitude, duration).
    - Calibrated initialization, measurement, and PSB threshold.
    - Native gate operations registered on qubit.xy channel.

State update:
    - Updates the x90/x180 pulse amplitude and duration via the
      XYDriveMacro.update() mechanism independently on each qubit.
    - The frequency detuning is optimised but not applied to the qubit
      state; the original intermediate frequency is preserved.
"""

node = QualibrationNode[CMAESGateParameters, Quam](
    name="03a_cmaes_gate_optimization",
    description=description,
    parameters=CMAESGateParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 5
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 50
    node.parameters.num_circuits = 5
    node.parameters.population_size = 10
    # node.parameters.simulate = True


node.machine = Quam.load()


# ── Helpers ──────────────────────────────────────────────────────────────

_PARAM_NAMES = [
    "amplitude_scale_target",
    "duration_offset_target",
    "freq_detuning_target",
    "amplitude_scale_control",
    "duration_offset_control",
    "freq_detuning_control",
]


def _quantize_duration(val: float, min_val: int = 16) -> int:
    """Quantize a continuous duration to a multiple of 4 ns, clamped."""
    return max(min_val, int(round(val / 4.0)) * 4)


def _generate_rb_circuits(
    num_circuits: int,
    depths: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Pre-generate all RB gate sequences for the given depths.

    Returns
    -------
    gates_flat, offsets, lengths
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
    """Play a single native gate with optional amplitude/duration override."""
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


def _fit_rb_decay_jax(depths: np.ndarray, survival: np.ndarray) -> np.ndarray:
    """Fit P(m) = A·p^m + B to batched survival data using JAX.

    Uses damped Gauss-Newton iteration.  Works for both decay directions
    (A > 0: decays down, A < 0: decays up).

    Parameters
    ----------
    depths : np.ndarray, shape (n_depths,)
        The RB circuit depths.
    survival : np.ndarray, shape (batch, n_depths)
        Survival probabilities for each batch element at each depth.

    Returns
    -------
    np.ndarray, shape (batch,)
        Clifford gate fidelity F = (1 + p) / 2 for each batch element.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial

    @partial(jax.vmap, in_axes=(None, 0))
    def _fit_single(m, surv):
        n = m.shape[0]

        b_init = surv[-1]
        a_init = surv[0] - b_init

        mid = n // 2
        ratio = jnp.where(
            jnp.abs(a_init) > 1e-6,
            jnp.clip((surv[mid] - b_init) / a_init, 1e-6, 1.0 - 1e-6),
            0.5,
        )
        p_init = jnp.power(ratio, 1.0 / m[mid])
        p_init = jnp.clip(p_init, 0.01, 0.9999)

        def refine_step(carry, _):
            a, p, b = carry
            p_m = jnp.power(p, m)
            pred = a * p_m + b
            residual = surv - pred

            dp_dm = a * p_m * jnp.log(jnp.clip(p, 1e-10, None))
            J = jnp.stack([p_m, dp_dm, jnp.ones(n)], axis=-1)

            # Levenberg-Marquardt damping for robustness
            lam = 1e-4 * jnp.trace(J.T @ J) + 1e-8
            JtJ = J.T @ J + lam * jnp.eye(3)
            Jtr = J.T @ residual
            delta = jnp.linalg.solve(JtJ, Jtr)

            a_new = a + delta[0]
            p_new = jnp.clip(p + delta[1], 1e-6, 1.0 - 1e-6)
            b_new = b + delta[2]
            return (a_new, p_new, b_new), None

        init_carry = (a_init, p_init, b_init)
        (_, p_final, _), _ = jax.lax.scan(
            refine_step, init_carry, None, length=15
        )

        fidelity = (1.0 + jnp.clip(p_final, 0.0, 1.0)) / 2.0
        return fidelity

    depths_jax = jnp.asarray(depths, dtype=jnp.float64)
    survival_jax = jnp.asarray(survival, dtype=jnp.float64)

    fidelities = _fit_single(depths_jax, survival_jax)
    return np.asarray(fidelities, dtype=np.float64)


def _compute_pair_fidelity(
    surv_target: np.ndarray,
    surv_control: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-qubit and average pair fidelity from RB survival data.

    Parameters
    ----------
    surv_target : np.ndarray, shape (pop_size, n_depths)
        Survival probabilities for qubit_target.
    surv_control : np.ndarray, shape (pop_size, n_depths)
        Survival probabilities for qubit_control.
    depths : np.ndarray, shape (n_depths,)
        The RB circuit depths.

    Returns
    -------
    scores : np.ndarray, shape (pop_size,)
        Mean fidelity across both qubits for each candidate.
    f_target : np.ndarray, shape (pop_size,)
        Per-candidate fidelity for qubit_target.
    f_control : np.ndarray, shape (pop_size,)
        Per-candidate fidelity for qubit_control.
    """
    stacked = np.concatenate([surv_target, surv_control], axis=0)
    all_fidelities = _fit_rb_decay_jax(depths, stacked)

    pop_size = surv_target.shape[0]
    f_target = all_fidelities[:pop_size]
    f_control = all_fidelities[pop_size:]

    f_target = np.where(np.isfinite(f_target), f_target, 0.5)
    f_control = np.where(np.isfinite(f_control), f_control, 0.5)

    scores = (f_target + f_control) / 2.0
    return scores, f_target, f_control


# %% {Create_QUA_program}
def _build_qua_program(node, qubit_pair, gates_flat, offsets, lengths, depths):
    """Build a QUA program for 5-point RB on both qubits in a pair.

    Each qubit receives its own amplitude_scale, duration, and frequency
    detuning via separate input streams.  The program runs RB sequentially
    on qubit_target then qubit_control for each candidate.

    Shape of one fetch per qubit: ``(pop_size, n_depths)``.
    """
    qubit_target = qubit_pair.qubit_target
    qubit_control = qubit_pair.qubit_control

    if_target = qubit_target.xy.intermediate_frequency
    if_control = qubit_control.xy.intermediate_frequency

    num_circuits = node.parameters.num_circuits
    num_shots = node.parameters.num_shots
    pop_size = node.parameters.population_size
    n_depths = len(depths)

    with program() as qua_prog:
        amp_target_in = declare_input_stream(
            "client", stream_id="amp_target", dtype=fixed, size=pop_size
        )
        dur_target_in = declare_input_stream(
            "client", stream_id="dur_target", dtype=int, size=pop_size
        )
        freq_det_target_in = declare_input_stream(
            "client", stream_id="freq_det_target", dtype=int, size=pop_size
        )
        amp_control_in = declare_input_stream(
            "client", stream_id="amp_control", dtype=fixed, size=pop_size
        )
        dur_control_in = declare_input_stream(
            "client", stream_id="dur_control", dtype=int, size=pop_size
        )
        freq_det_control_in = declare_input_stream(
            "client", stream_id="freq_det_control", dtype=int, size=pop_size
        )

        amp_scale_t = declare(fixed)
        dur_t = declare(int)
        freq_det_t = declare(int)
        amp_scale_c = declare(fixed)
        dur_c = declare(int)
        freq_det_c = declare(int)

        gates_qua = declare(int, value=gates_flat)
        offsets_qua = declare(int, value=offsets)
        lengths_qua = declare(int, value=lengths)

        candidate_idx = declare(int)
        depth_idx = declare(int)
        circuit_idx = declare(int)
        shot_idx = declare(int)
        gate_idx = declare(int)
        current_gate = declare(int)
        seq_index = declare(int)
        seq_offset = declare(int)
        seq_length = declare(int)

        state_target = declare(int)
        state_control = declare(int)
        state_target_st = declare_output_stream()
        state_control_st = declare_output_stream()

        with infinite_loop_():
            advance_input_stream(amp_target_in)
            advance_input_stream(dur_target_in)
            advance_input_stream(freq_det_target_in)
            advance_input_stream(amp_control_in)
            advance_input_stream(dur_control_in)
            advance_input_stream(freq_det_control_in)
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                assign(amp_scale_t, amp_target_in[candidate_idx])
                assign(dur_t, dur_target_in[candidate_idx])
                assign(freq_det_t, freq_det_target_in[candidate_idx])
                assign(amp_scale_c, amp_control_in[candidate_idx])
                assign(dur_c, dur_control_in[candidate_idx])
                assign(freq_det_c, freq_det_control_in[candidate_idx])

                # --- RB on qubit_target ---
                qubit_target.xy.update_frequency(if_target + freq_det_t)
                with for_(depth_idx, 0, depth_idx < n_depths, depth_idx + 1):
                    with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                        assign(seq_index, depth_idx * num_circuits + circuit_idx)
                        assign(seq_offset, offsets_qua[seq_index])
                        assign(seq_length, lengths_qua[seq_index])

                        with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                            reset_frame(qubit_target.xy.name)
                            align()
                            qubit_target.empty()
                            qubit_target.initialize(
                                conditional_drive=True,
                            )
                            align()

                            with for_(gate_idx, 0, gate_idx < seq_length, gate_idx + 1):
                                assign(current_gate, gates_qua[seq_offset + gate_idx])
                                _play_rb_gate_scaled(qubit_target, current_gate, amp_scale_t, dur_t)

                            align()
                            p = qubit_target.measure()
                            align()
                            qubit_target.voltage_sequence.ramp_to_zero()
                            align()

                            assign(state_target, Cast.to_int(p))
                            save(state_target, state_target_st)

                qubit_target.xy.update_frequency(if_target)

                # --- RB on qubit_control ---
                qubit_control.xy.update_frequency(if_control + freq_det_c)
                with for_(depth_idx, 0, depth_idx < n_depths, depth_idx + 1):
                    with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                        assign(seq_index, depth_idx * num_circuits + circuit_idx)
                        assign(seq_offset, offsets_qua[seq_index])
                        assign(seq_length, lengths_qua[seq_index])

                        with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                            reset_frame(qubit_control.xy.name)
                            align()
                            qubit_control.empty()
                            qubit_control.initialize(
                                conditional_drive=True,
                            )
                            align()

                            with for_(gate_idx, 0, gate_idx < seq_length, gate_idx + 1):
                                assign(current_gate, gates_qua[seq_offset + gate_idx])
                                _play_rb_gate_scaled(qubit_control, current_gate, amp_scale_c, dur_c)

                            align()
                            p = qubit_control.measure()
                            align()
                            qubit_control.voltage_sequence.ramp_to_zero()
                            align()

                            assign(state_control, Cast.to_int(p))
                            save(state_control, state_control_st)

                qubit_control.xy.update_frequency(if_control)

        with stream_processing():
            (
                state_target_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(n_depths)
                .buffer(pop_size)
                .save_all("survival_target")
            )
            (
                state_control_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(n_depths)
                .buffer(pop_size)
                .save_all("survival_control")
            )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Validate parameters, compute the RB depths, and compile the QUA program."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    if not qubit_pairs:
        raise ValueError("No qubit pairs resolved — check qubit_pairs parameter or machine config.")

    n_depths = node.parameters.num_rb_depths
    depths = np.unique(
        np.round(np.geomspace(node.parameters.depth_min, node.parameters.depth_max, n_depths))
        .astype(int)
    ).tolist()
    if len(depths) < n_depths:
        node.log(
            f"  Warning: only {len(depths)} unique depths after rounding "
            f"(requested {n_depths}). Consider widening depth_min/depth_max."
        )
    n_depths = len(depths)
    node.namespace["depths"] = depths

    rng = np.random.default_rng(node.parameters.seed)

    qp = qubit_pairs[0]
    gates_flat, offsets, lengths = _generate_rb_circuits(
        node.parameters.num_circuits, depths, rng,
    )
    node.namespace["rb_circuits"] = {
        "gates_flat": gates_flat,
        "offsets": offsets,
        "lengths": lengths,
    }

    node.namespace["qua_program"] = _build_qua_program(
        node, qp, gates_flat, offsets, lengths, depths,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[CMAESGateParameters, Quam]):
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


# %% {Run_CMA-ES_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_cmaes_loop(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    The 6D search space is:
        [amp_scale_target, dur_offset_target, freq_detuning_target,
         amp_scale_control, dur_offset_control, freq_detuning_control]

    For each pair, a single QUA program runs RB on both qubit_target and
    qubit_control with independent pulse parameters.  The score is the
    average fidelity across both qubits, estimated from a 5-point
    exponential decay fit.
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]
    depths = node.namespace["depths"]

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    pop_size = node.parameters.population_size

    lo = np.array([
        node.parameters.amplitude_scale_min,
        node.parameters.duration_offset_min,
        node.parameters.freq_detuning_min,
        node.parameters.amplitude_scale_min,
        node.parameters.duration_offset_min,
        node.parameters.freq_detuning_min,
    ])
    hi = np.array([
        node.parameters.amplitude_scale_max,
        node.parameters.duration_offset_max,
        node.parameters.freq_detuning_max,
        node.parameters.amplitude_scale_max,
        node.parameters.duration_offset_max,
        node.parameters.freq_detuning_max,
    ])
    param_range = hi - lo

    x0_phys = np.array([
        node.parameters.amplitude_scale_initial,
        node.parameters.duration_offset_initial,
        node.parameters.freq_detuning_initial,
        node.parameters.amplitude_scale_initial,
        node.parameters.duration_offset_initial,
        node.parameters.freq_detuning_initial,
    ])
    x0_norm = (x0_phys - lo) / param_range
    bounds_norm = [(0.0, 1.0)] * len(_PARAM_NAMES)

    depths_arr = np.array(depths, dtype=np.float64)

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES gate optimisation for pair {qp.name}...")

        qubit_target = qp.qubit_target
        qubit_control = qp.qubit_control

        rng = np.random.default_rng(node.parameters.seed)
        gates_flat, offsets, lengths = _generate_rb_circuits(
            node.parameters.num_circuits, depths, rng,
        )

        qua_prog = _build_qua_program(
            node, qp, gates_flat, offsets, lengths, depths,
        )

        cal_dur_target = qubit_target.macros["x90"].pulse.length
        cal_dur_control = qubit_control.macros["x90"].pulse.length

        survival_streams = {
            f"target_depth_{i}": [] for i in range(len(depths))
        }
        survival_streams.update({
            f"control_depth_{i}": [] for i in range(len(depths))
        })
        fidelity_history = {
            "target": [], "control": [], "average": [],
            "all_target": [], "all_control": [], "all_average": [],
            "best_surv_target": [],
            "best_surv_control": [],
            "running_best_average": [],
            "running_best_target": [],
            "running_best_control": [],
            "running_best_surv_target": [],
            "running_best_surv_control": [],
        }

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            target_handle = job.result_handles.get("survival_target")
            control_handle = job.result_handles.get("survival_control")
            generation_counter = 0

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                """Push one generation and compute average pair fidelity.

                candidates_norm columns:
                    0: amp_scale_target (normalised)
                    1: dur_offset_target (normalised)
                    2: freq_detuning_target (normalised)
                    3: amp_scale_control (normalised)
                    4: dur_offset_control (normalised)
                    5: freq_detuning_control (normalised)
                """
                nonlocal generation_counter

                candidates_phys = lo + candidates_norm * param_range
                job.push_to_input_stream(
                    "amp_target", [float(c[0]) for c in candidates_phys]
                )
                job.push_to_input_stream(
                    "dur_target",
                    [_quantize_duration(cal_dur_target + c[1]) for c in candidates_phys],
                )
                job.push_to_input_stream(
                    "freq_det_target", [int(round(c[2])) for c in candidates_phys]
                )
                job.push_to_input_stream(
                    "amp_control", [float(c[3]) for c in candidates_phys]
                )
                job.push_to_input_stream(
                    "dur_control",
                    [_quantize_duration(cal_dur_control + c[4]) for c in candidates_phys],
                )
                job.push_to_input_stream(
                    "freq_det_control", [int(round(c[5])) for c in candidates_phys]
                )

                target_count = generation_counter + 1
                while target_handle.count_so_far() < target_count:
                    _time.sleep(0.005)
                while control_handle.count_so_far() < target_count:
                    _time.sleep(0.005)

                surv_target = np.asarray(
                    target_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
                surv_control = np.asarray(
                    control_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
                generation_counter += 1

                for i in range(len(depths)):
                    survival_streams[f"target_depth_{i}"].append(surv_target[:, i].copy())
                    survival_streams[f"control_depth_{i}"].append(surv_control[:, i].copy())

                scores, ft, fc = _compute_pair_fidelity(
                    surv_target, surv_control, depths_arr
                )

                best_idx = int(np.argmax(scores))
                fidelity_history["target"].append(float(ft[best_idx]))
                fidelity_history["control"].append(float(fc[best_idx]))
                fidelity_history["average"].append(float(scores[best_idx]))

                fidelity_history["best_surv_target"].append(surv_target[best_idx].copy())
                fidelity_history["best_surv_control"].append(surv_control[best_idx].copy())

                fidelity_history["all_target"].append(ft.copy())
                fidelity_history["all_control"].append(fc.copy())
                fidelity_history["all_average"].append(scores.copy())

                current_best = float(scores[best_idx])
                prev_best = (
                    fidelity_history["running_best_average"][-1]
                    if fidelity_history["running_best_average"]
                    else -1.0
                )
                if current_best >= prev_best:
                    fidelity_history["running_best_average"].append(current_best)
                    fidelity_history["running_best_target"].append(float(ft[best_idx]))
                    fidelity_history["running_best_control"].append(float(fc[best_idx]))
                    fidelity_history["running_best_surv_target"].append(
                        surv_target[best_idx].copy()
                    )
                    fidelity_history["running_best_surv_control"].append(
                        surv_control[best_idx].copy()
                    )
                else:
                    fidelity_history["running_best_average"].append(prev_best)
                    fidelity_history["running_best_target"].append(
                        fidelity_history["running_best_target"][-1]
                    )
                    fidelity_history["running_best_control"].append(
                        fidelity_history["running_best_control"][-1]
                    )
                    fidelity_history["running_best_surv_target"].append(
                        fidelity_history["running_best_surv_target"][-1]
                    )
                    fidelity_history["running_best_surv_control"].append(
                        fidelity_history["running_best_surv_control"][-1]
                    )

                return scores

            try:
                opt_result = run_cmaes_optimization(
                    evaluate_fn=evaluate_candidates,
                    param_names=_PARAM_NAMES,
                    x0=x0_norm,
                    sigma0=node.parameters.sigma0,
                    bounds=bounds_norm,
                    population_size=pop_size,
                    max_generations=node.parameters.max_generations,
                    tolx=node.parameters.tolx,
                    tolfun=node.parameters.tolfun,
                    log_callable=node.log,
                    progress_prefix=qp.name,
                    log_each_generation=node.parameters.cmaes_log_each_generation,
                )
                opt_result.best_params = lo + opt_result.best_params * param_range
                opt_result.param_history = [
                    lo + h * param_range for h in opt_result.param_history
                ]
                opt_result.all_candidates = [
                    lo + c * param_range for c in opt_result.all_candidates
                ]
                optimization_results[qp.name] = opt_result
                measurement_streams[qp.name] = survival_streams
                measurement_streams[qp.name]["fidelity_history"] = fidelity_history
            finally:
                job.cancel()

    node.namespace["optimization_results"] = optimization_results
    node.namespace["measurement_streams"] = measurement_streams
    node.results["optimization_results"] = {
        name: result.to_dict() for name, result in optimization_results.items()
    }
    def _serialize_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: _serialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            return [arr.tolist() for arr in v]
        return v

    node.results["measurement_streams"] = {
        name: _serialize_value(streams)
        for name, streams in measurement_streams.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["optimization_results"] = {
        name: OptimizationResult.from_dict(d)
        for name, d in node.results["optimization_results"].items()
    }
    def _deserialize_value(v):
        if isinstance(v, dict):
            return {kk: _deserialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return [np.array(arr) for arr in v]
        return v

    raw_streams = node.results.get("measurement_streams", {})
    node.namespace["measurement_streams"] = {
        name: _deserialize_value(streams)
        for name, streams in raw_streams.items()
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Summarise the CMA-ES optimisation outcome per qubit pair.

    Computes per-qubit fidelities at the best candidate and logs
    both individual and average fidelities.
    """
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )

    for qp_name, summary in fit_results.items():
        streams = measurement_streams.get(qp_name, {})
        fid_hist = streams.get("fidelity_history", {})
        if fid_hist:
            summary["best_fidelity_target"] = max(fid_hist["target"])
            summary["best_fidelity_control"] = max(fid_hist["control"])
            summary["best_fidelity_average"] = max(fid_hist["average"])
            summary["final_fidelity_target"] = fid_hist["target"][-1]
            summary["final_fidelity_control"] = fid_hist["control"][-1]

    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    for qp in node.namespace["qubit_pairs"]:
        summary = fit_results.get(qp.name)
        if summary is None:
            continue
        f_t = summary.get("best_fidelity_target")
        f_c = summary.get("best_fidelity_control")
        f_avg = summary.get("best_fidelity_average")
        if f_t is not None:
            node.log(
                f"  [{qp.name}] Per-qubit best fidelities: "
                f"{qp.qubit_target.name}={f_t:.6f}, "
                f"{qp.qubit_control.name}={f_c:.6f}, "
                f"average={f_avg:.6f}"
            )

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qp_name, summary in fit_results.items()
    }


# %% {Plot_data}
def _plot_rb_survival_streams_on_ax(
    ax: plt.Axes,
    fidelity_history: dict,
    n_depths: int,
    depths: list | np.ndarray,
    pair_name: str = "",
) -> None:
    """Plot the best candidate's survival probability at each depth vs generation."""
    best_surv_t = fidelity_history.get("best_surv_target", [])
    best_surv_c = fidelity_history.get("best_surv_control", [])
    if not best_surv_t and not best_surv_c:
        ax.set_title(f"No best-candidate survival data — {pair_name}")
        return

    colors_target = [f"C{i}" for i in range(n_depths)]
    colors_control = [f"C{i + n_depths}" for i in range(n_depths)]

    for i in range(n_depths):
        depth_label = depths[i] if i < len(depths) else i

        if best_surv_t:
            best_vals = np.array([np.asarray(s)[i] for s in best_surv_t])
            ax.plot(
                np.arange(1, len(best_vals) + 1), best_vals,
                "o-", color=colors_target[i], markersize=4, linewidth=1.5,
                label=f"T m={depth_label}",
            )

        if best_surv_c:
            best_vals = np.array([np.asarray(s)[i] for s in best_surv_c])
            ax.plot(
                np.arange(1, len(best_vals) + 1), best_vals,
                "s--", color=colors_control[i], markersize=4, linewidth=1.5,
                label=f"C m={depth_label}",
            )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival probability")
    title = (
        f"Best-candidate RB survival — {pair_name}" if pair_name
        else "Best-candidate RB survival"
    )
    ax.set_title(title)
    ax.legend(fontsize="x-small", ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_per_qubit_fidelity_on_ax(
    ax: plt.Axes,
    fidelity_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot per-qubit and average fidelity convergence vs generation."""
    if not fidelity_history:
        ax.set_title(f"No fidelity data — {pair_name}")
        return

    n_gen = len(fidelity_history["average"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, fidelity_history["target"], "o-",
        color="C0", markersize=4, label=qubit_target_name,
    )
    ax.plot(
        generations, fidelity_history["control"], "s-",
        color="C1", markersize=4, label=qubit_control_name,
    )
    ax.plot(
        generations, fidelity_history["average"], "D-",
        color="C2", markersize=4, label="Average",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Clifford gate fidelity")
    title = (
        f"Per-qubit fidelity — {pair_name}" if pair_name
        else "Per-qubit fidelity"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_individual_fidelities_on_ax(
    ax: plt.Axes,
    fidelity_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Scatter all candidate fidelities per generation with running best overlay."""
    if not fidelity_history or not fidelity_history.get("all_average"):
        ax.set_title(f"No individual fidelity data — {pair_name}")
        return

    n_gen = len(fidelity_history["all_average"])
    for gen_idx in range(n_gen):
        gen_num = gen_idx + 1
        avg_vals = np.asarray(fidelity_history["all_average"][gen_idx])
        ax.scatter(
            np.full_like(avg_vals, gen_num, dtype=float), avg_vals,
            s=10, alpha=0.3, color="C7", zorder=1,
        )

    generations = np.arange(1, n_gen + 1)
    if fidelity_history.get("running_best_average"):
        ax.plot(
            generations, fidelity_history["running_best_average"],
            "D-", color="C2", markersize=5, linewidth=2,
            label="Running best (avg)", zorder=3,
        )
    if fidelity_history.get("running_best_target"):
        ax.plot(
            generations, fidelity_history["running_best_target"],
            "o--", color="C0", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_target_name})", zorder=2,
        )
    if fidelity_history.get("running_best_control"):
        ax.plot(
            generations, fidelity_history["running_best_control"],
            "s--", color="C1", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_control_name})", zorder=2,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Clifford gate fidelity")
    title = (
        f"Individual fidelities & running best — {pair_name}" if pair_name
        else "Individual fidelities & running best"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_running_best_rb_curves_on_ax(
    ax: plt.Axes,
    fidelity_history: dict,
    depths: list | np.ndarray,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot RB survival decay curves for the final running-best candidate."""
    surv_t_list = fidelity_history.get("running_best_surv_target", [])
    surv_c_list = fidelity_history.get("running_best_surv_control", [])
    if not surv_t_list or not surv_c_list:
        ax.set_title(f"No running-best RB curves — {pair_name}")
        return

    surv_t = np.asarray(surv_t_list[-1])
    surv_c = np.asarray(surv_c_list[-1])
    depths_arr = np.asarray(depths)

    ax.plot(
        depths_arr, surv_t, "o-", color="C0", markersize=6,
        label=qubit_target_name,
    )
    ax.plot(
        depths_arr, surv_c, "s-", color="C1", markersize=6,
        label=qubit_control_name,
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    rb_avg = fidelity_history.get("running_best_average", [])
    if rb_avg:
        ax.text(
            0.98, 0.02,
            f"Best avg fidelity: {rb_avg[-1]:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    ax.set_xlabel("Clifford sequence depth")
    ax.set_ylabel("Survival probability")
    title = (
        f"Running-best RB decay — {pair_name}" if pair_name
        else "Running-best RB decay"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Generate convergence, per-qubit fidelity, and parameter-evolution plots."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    pair_names = list(opt_results.keys())
    n_pairs = max(len(pair_names), 1)
    n_depths = node.parameters.num_rb_depths

    depths = np.unique(
        np.round(np.geomspace(node.parameters.depth_min, node.parameters.depth_max, n_depths))
        .astype(int)
    ).tolist()

    fig_combined, axes = plt.subplots(
        5, n_pairs, figsize=(9 * n_pairs, 22), squeeze=False,
    )

    for col, pname in enumerate(pair_names):
        streams = measurement_streams.get(pname, {})
        fid_hist = streams.get("fidelity_history", {})

        _plot_rb_survival_streams_on_ax(
            axes[0, col], fid_hist, len(depths), depths, pname,
        )
        qp = node.namespace["qubit_pairs"][col]
        _plot_per_qubit_fidelity_on_ax(
            axes[1, col], fid_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )

        plot_score_convergence_on_ax(axes[2, col], opt_results[pname], pname)

        _plot_individual_fidelities_on_ax(
            axes[3, col], fid_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )

        _plot_running_best_rb_curves_on_ax(
            axes[4, col], fid_hist, depths,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )

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
def update_state(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Apply optimal per-qubit x90 amplitude and duration."""
    fit_results = node.results["fit_results"]

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            pair_summary = fit_results.get(qp.name)
            if pair_summary is None:
                continue
            if not pair_summary["success"]:
                node.log(
                    f"  {qp.name}: optimisation did not succeed — skipping."
                )
                continue

            best = pair_summary["best_params"]
            f_target = pair_summary.get("best_fidelity_target", pair_summary["best_score"])
            f_control = pair_summary.get("best_fidelity_control", pair_summary["best_score"])

            qubit_params = [
                (qp.qubit_target, best["amplitude_scale_target"], best["duration_offset_target"], best["freq_detuning_target"], f_target),
                (qp.qubit_control, best["amplitude_scale_control"], best["duration_offset_control"], best["freq_detuning_control"], f_control),
            ]

            for qubit, opt_amp_scale, opt_dur_offset, opt_freq_det, fidelity in qubit_params:
                xy_macro = qubit.macros["x90"]

                current_x90_amp = xy_macro.pulse.amplitude
                current_pi_amp = xy_macro.pi_pulse.amplitude
                current_duration = xy_macro.pulse.length
                current_larmor = qubit.larmor_frequency

                new_duration = _quantize_duration(current_duration + opt_dur_offset)
                # Larmor flows into the (integer) drive IF = larmor − LO, so keep it int.
                new_larmor = int(round(current_larmor + opt_freq_det))

                xy_macro.update(
                    amplitude_scale=opt_amp_scale,
                    duration=new_duration,
                )
                qubit.larmor_frequency = new_larmor

                node.log(
                    f"  {qp.name}/{qubit.name}: gate params updated — "
                    f"x90_amp: {current_x90_amp:.6g} → {current_x90_amp * opt_amp_scale:.6g} V, "
                    f"x180_amp: {current_pi_amp:.6g} → {current_pi_amp * opt_amp_scale:.6g} V, "
                    f"duration: {current_duration} → {new_duration} ns, "
                    f"freq_detuning: {opt_freq_det:+.0f} Hz, "
                    f"larmor: {current_larmor:.0f} → {new_larmor:.0f} Hz, "
                    f"fidelity={fidelity:.6f}"
                )

            node.log(
                f"  {qp.name}: average pair fidelity = {pair_summary['best_score']:.6f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CMAESGateParameters, Quam]):
    """Persist all results, figures, and parameters to disk."""
    node.save()
