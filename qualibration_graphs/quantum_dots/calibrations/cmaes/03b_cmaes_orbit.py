"""CMA-ES single-qubit gate optimisation using orbit separation scoring.

Overview
--------
This node optimises the x90 pulse amplitude, duration, and drive frequency
independently for each qubit in a pair, maximising an orbit separation score
that quantifies gate quality without requiring a multi-depth RB sweep.

For each qubit in the pair, two survival probabilities are measured at a
single fixed Clifford depth:

    P_normal:  initialize |0⟩ → random Clifford sequence → measure
    P_pi:      initialize |0⟩ → π-pulse → random Clifford sequence → measure

Each variant uses independent random circuit instances to avoid correlations.
The score is maximised separation:

    score_qubit = P_normal − P_pi

For perfect single-qubit gates this equals p^m where p = 2F − 1 is the
depolarising parameter and m is the orbit depth.  The per-pair score is:

    score = (score_target + score_control) / 2

Optimal orbit depth
-------------------
The depth is chosen to maximise the Fisher information of the orbit score
with respect to gate fidelity.  The sensitivity condition:

    1 + m·ln(p) = p^(2m)

yields the closed-form approximation (valid for small r = 1−p):

    m* ≈ 0.8 / (2·(1 − F))

For the target fidelity F = 99% this gives m* = 40, at which the expected
score is p^40 = 0.98^40 ≈ 0.45 — well above the noise floor while
retaining maximum gradient for CMA-ES to exploit.

Algorithm
---------
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) maintains a
multivariate Gaussian distribution over six parameters:

    θ = [amplitude_scale_target, duration_offset_target, freq_detuning_target,
         amplitude_scale_control, duration_offset_control, freq_detuning_control]

At each generation, CMA-ES:

1. **Samples** ``population_size`` candidate parameter vectors.
2. **Evaluates** all candidates in a single QUA program execution.
3. **Updates** μ, σ, and C via the CMA-ES adaptation rules.

QUA program architecture
------------------------
For each candidate and each qubit:

    for circuit in range(num_circuits):
        for shot in range(num_shots):
            — Variant 1 (normal): initialize → play sequence → measure
            — Variant 2 (pi):     initialize → x180 → play sequence → measure

Stream processing produces arrays of shape ``(pop_size, 2)`` per qubit per
generation, where the two columns are [P_normal, P_pi].

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
from calibration_utils.cmaes.cmaes_orbit_parameters import CMAESOrbitParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_qubit_pairs
from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    NUM_CLIFFORDS,
    build_single_qubit_clifford_tables,
)
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
        CMA-ES GATE OPTIMISATION — ORBIT SEPARATION (QUBIT-PAIR, PER-QUBIT PARAMS)
Uses CMA-ES to optimise single-qubit gate parameters (x90 amplitude scale,
duration offset, and frequency detuning) independently for each qubit in a
pair, maximising the orbit separation score.

The orbit score measures the difference in survival probability between two
preparations at a fixed Clifford depth m:
    score = P(init=|0⟩, depth=m) − P(init=|1⟩, depth=m)
         ≈ p^m  (depolarising parameter to the m-th power)

This is equivalent to the information in a full RB curve but measured at a
single depth, making it faster to evaluate.

The search space is 6-dimensional:
    [amp_scale_target, dur_offset_target, freq_detuning_target,
     amp_scale_control, dur_offset_control, freq_detuning_control]

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

node = QualibrationNode[CMAESOrbitParameters, Quam](
    name="03b_cmaes_orbit",
    description=description,
    parameters=CMAESOrbitParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 100
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 500
    node.parameters.num_circuits = 20
    node.parameters.population_size = 10
    node.parameters.orbit_depth = 30
    # node.parameters.simulate = True


node.machine = Quam.load()

_CLIFFORD_TABLES = build_single_qubit_clifford_tables()

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


def _generate_orbit_circuits(
    num_circuits: int,
    depth: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pre-generate random Clifford index sequences for orbit measurement (no recovery).

    Returns
    -------
    cliffords_flat
        Flattened Clifford indices, row-major with stride ``depth``.
        Circuit ``i`` occupies ``cliffords_flat[i*depth : (i+1)*depth]``.
        Decomposition into native gates is done on-chip via the Clifford tables.
    """
    cliffords_flat = []
    for _ in range(num_circuits):
        cliffords_flat.extend(rng.integers(0, NUM_CLIFFORDS, size=depth).tolist())
    return cliffords_flat


def _play_gate_scaled(qubit, gate_int, amplitude_scale=None, duration=None):
    """Play a single native gate with optional amplitude/duration override.

    Gate integers 0-5 match the alternative decomposition (physical gates only).
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


# %% {Create_QUA_program}
def _build_qua_program(
    node, qubit_pair, depth,
    cliffords_normal_flat, cliffords_pi_flat,
):
    """Build a QUA program for orbit measurement on both qubits in a pair.

    For each qubit and candidate, runs two variants:
      1. Normal init → random sequence → measure
      2. Normal init → x180 → random sequence → measure

    Each variant uses independent pre-generated random circuits.
    Shape of one fetch per qubit: ``(pop_size, 2)`` where columns are
    [P_normal, P_pi].
    """
    qubit_target = qubit_pair.qubit_target
    qubit_control = qubit_pair.qubit_control

    if_target = qubit_target.xy.intermediate_frequency
    if_control = qubit_control.xy.intermediate_frequency

    num_circuits = node.parameters.num_circuits
    num_shots = node.parameters.num_shots
    pop_size = node.parameters.population_size

    with program() as qua_prog:
        amp_target_in = declare_input_stream(
            "client", stream_id="amp_target", dtype=fixed
        )
        dur_target_in = declare_input_stream(
            "client", stream_id="dur_target", dtype=int
        )
        freq_det_target_in = declare_input_stream(
            "client", stream_id="freq_det_target", dtype=int
        )
        amp_control_in = declare_input_stream(
            "client", stream_id="amp_control", dtype=fixed
        )
        dur_control_in = declare_input_stream(
            "client", stream_id="dur_control", dtype=int
        )
        freq_det_control_in = declare_input_stream(
            "client", stream_id="freq_det_control", dtype=int
        )

        amp_scale_t = declare(fixed)
        dur_t = declare(int)
        freq_det_t = declare(int)
        amp_scale_c = declare(fixed)
        dur_c = declare(int)
        freq_det_c = declare(int)

        # Clifford index sequences; circuit i occupies [i*depth : (i+1)*depth]
        cliffords_normal_qua = declare(int, value=cliffords_normal_flat)
        cliffords_pi_qua = declare(int, value=cliffords_pi_flat)

        # Clifford decomposition lookup tables (alternative decomposition)
        clifford_decomp_qua = declare(int, value=_CLIFFORD_TABLES["decomp_flat"])
        clifford_decomp_offsets_qua = declare(int, value=_CLIFFORD_TABLES["decomp_offsets"])
        clifford_decomp_lengths_qua = declare(int, value=_CLIFFORD_TABLES["decomp_lengths"])

        candidate_idx = declare(int)
        circuit_idx = declare(int)
        shot_idx = declare(int)
        cliff_loop_idx = declare(int)
        gate_idx = declare(int)
        rand_clifford = declare(int)
        decomp_offset = declare(int)
        decomp_length = declare(int)
        current_gate = declare(int)

        state_target = declare(int)
        state_control = declare(int)
        state_target_st = declare_output_stream()
        state_control_st = declare_output_stream()

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(amp_target_in)
                advance_input_stream(dur_target_in)
                advance_input_stream(freq_det_target_in)
                advance_input_stream(amp_control_in)
                advance_input_stream(dur_control_in)
                advance_input_stream(freq_det_control_in)
                assign(amp_scale_t, amp_target_in)
                assign(dur_t, dur_target_in)
                assign(freq_det_t, freq_det_target_in)
                assign(amp_scale_c, amp_control_in)
                assign(dur_c, dur_control_in)
                assign(freq_det_c, freq_det_control_in)

                # --- Orbit on qubit_target ---
                qubit_target.xy.update_frequency(if_target + freq_det_t)

                # Variant 1: normal initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_target.xy.name)
                        align()
                        qubit_target.initialize(
                            conditional_drive=True,
                        )
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_normal_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_target, current_gate, amp_scale_t, dur_t)

                        align()
                        p = qubit_target.measure()
                        align()
                        qubit_target.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_target, Cast.to_int(p))
                        save(state_target, state_target_st)

                # Variant 2: pi-pulse initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_target.xy.name)
                        align()
                        qubit_target.initialize(
                            conditional_drive=True,
                        )
                        align()
                        qubit_target.x180(amplitude_scale=amp_scale_t, duration=dur_t)
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_pi_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_target, current_gate, amp_scale_t, dur_t)

                        align()
                        p = qubit_target.measure()
                        align()
                        qubit_target.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_target, Cast.to_int(p))
                        save(state_target, state_target_st)

                qubit_target.xy.update_frequency(if_target)

                # --- Orbit on qubit_control ---
                qubit_control.xy.update_frequency(if_control + freq_det_c)

                # Variant 1: normal initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_control.xy.name)
                        align()
                        qubit_control.initialize(
                            conditional_drive=True,
                        )
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_normal_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_control, current_gate, amp_scale_c, dur_c)

                        align()
                        p = qubit_control.measure()
                        align()
                        qubit_control.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_control, Cast.to_int(p))
                        save(state_control, state_control_st)

                # Variant 2: pi-pulse initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_control.xy.name)
                        align()
                        qubit_control.initialize(
                            conditional_drive=True,
                        )
                        align()
                        qubit_control.x180(amplitude_scale=amp_scale_c, duration=dur_c)
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_pi_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_control, current_gate, amp_scale_c, dur_c)

                        align()
                        p = qubit_control.measure()
                        align()
                        qubit_control.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_control, Cast.to_int(p))
                        save(state_control, state_control_st)

                qubit_control.xy.update_frequency(if_control)

        with stream_processing():
            # Per qubit: num_circuits shots for normal, then num_circuits for pi
            # → buffer(num_shots) averages shots, buffer(num_circuits) averages
            #   circuits, buffer(2) groups [normal, pi], buffer(pop_size) groups
            #   candidates.
            (
                state_target_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(2)
                .buffer(pop_size)
                .save_all("survival_target")
            )
            (
                state_control_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(2)
                .buffer(pop_size)
                .save_all("survival_control")
            )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Validate parameters, generate orbit circuits, and compile the QUA program."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    if not qubit_pairs:
        raise ValueError("No qubit pairs resolved — check qubit_pairs parameter or machine config.")

    depth = node.parameters.orbit_depth
    node.namespace["orbit_depth"] = depth

    rng = np.random.default_rng(node.parameters.seed)

    cliffords_normal_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
    cliffords_pi_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
    node.namespace["orbit_circuits"] = {
        "cliffords_normal_flat": cliffords_normal_flat,
        "cliffords_pi_flat": cliffords_pi_flat,
    }

    qp = qubit_pairs[0]
    node.namespace["qua_program"] = _build_qua_program(
        node, qp, depth, cliffords_normal_flat, cliffords_pi_flat,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[CMAESOrbitParameters, Quam]):
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
def run_cmaes_loop(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    The score is the orbit separation: P_normal − P_pi, averaged across
    both qubits in the pair.
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]
    depth = node.namespace["orbit_depth"]

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

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES orbit optimisation for pair {qp.name}...")

        qubit_target = qp.qubit_target
        qubit_control = qp.qubit_control

        rng = np.random.default_rng(node.parameters.seed)
        cliffords_normal_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
        cliffords_pi_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)

        qua_prog = _build_qua_program(
            node, qp, depth, cliffords_normal_flat, cliffords_pi_flat,
        )

        cal_dur_target = qubit_target.macros["x90"].pulse.length
        cal_dur_control = qubit_control.macros["x90"].pulse.length

        orbit_history = {
            "p_normal_target": [], "p_pi_target": [],
            "p_normal_control": [], "p_pi_control": [],
            "separation_target": [], "separation_control": [],
            "score_average": [],
            "all_separation_target": [], "all_separation_control": [],
            "all_score_average": [],
            "running_best_score": [],
            "running_best_sep_target": [],
            "running_best_sep_control": [],
            "running_best_p_normal_target": [],
            "running_best_p_pi_target": [],
            "running_best_p_normal_control": [],
            "running_best_p_pi_control": [],
        }

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            target_handle = job.result_handles.get("survival_target")
            control_handle = job.result_handles.get("survival_control")
            generation_counter = 0

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                """Push one generation and compute orbit separation scores."""
                nonlocal generation_counter

                candidates_phys = lo + candidates_norm * param_range
                for c in candidates_phys:
                    job.push_to_input_stream("amp_target", float(c[0]))
                    job.push_to_input_stream(
                        "dur_target", _quantize_duration(cal_dur_target + c[1])
                    )
                    job.push_to_input_stream("freq_det_target", int(round(c[2])))
                    job.push_to_input_stream("amp_control", float(c[3]))
                    job.push_to_input_stream(
                        "dur_control", _quantize_duration(cal_dur_control + c[4])
                    )
                    job.push_to_input_stream("freq_det_control", int(round(c[5])))

                target_count = generation_counter + 1
                while target_handle.count_so_far() < target_count:
                    _time.sleep(0.005)
                while control_handle.count_so_far() < target_count:
                    _time.sleep(0.005)

                # Shape: (pop_size, 2) where [:, 0] = normal, [:, 1] = pi
                surv_target = np.asarray(
                    target_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
                surv_control = np.asarray(
                    control_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
                generation_counter += 1

                sep_target = np.abs(surv_target[:, 0] - surv_target[:, 1])
                sep_control = np.abs(surv_control[:, 0] - surv_control[:, 1])
                scores = (sep_target + sep_control) / 2.0

                best_idx = int(np.argmax(scores))
                orbit_history["p_normal_target"].append(float(surv_target[best_idx, 0]))
                orbit_history["p_pi_target"].append(float(surv_target[best_idx, 1]))
                orbit_history["p_normal_control"].append(float(surv_control[best_idx, 0]))
                orbit_history["p_pi_control"].append(float(surv_control[best_idx, 1]))
                orbit_history["separation_target"].append(float(sep_target[best_idx]))
                orbit_history["separation_control"].append(float(sep_control[best_idx]))
                orbit_history["score_average"].append(float(scores[best_idx]))

                orbit_history["all_separation_target"].append(sep_target.copy())
                orbit_history["all_separation_control"].append(sep_control.copy())
                orbit_history["all_score_average"].append(scores.copy())

                current_best = float(scores[best_idx])
                prev_best = (
                    orbit_history["running_best_score"][-1]
                    if orbit_history["running_best_score"]
                    else -1.0
                )
                if current_best >= prev_best:
                    orbit_history["running_best_score"].append(current_best)
                    orbit_history["running_best_sep_target"].append(float(sep_target[best_idx]))
                    orbit_history["running_best_sep_control"].append(float(sep_control[best_idx]))
                    orbit_history["running_best_p_normal_target"].append(float(surv_target[best_idx, 0]))
                    orbit_history["running_best_p_pi_target"].append(float(surv_target[best_idx, 1]))
                    orbit_history["running_best_p_normal_control"].append(float(surv_control[best_idx, 0]))
                    orbit_history["running_best_p_pi_control"].append(float(surv_control[best_idx, 1]))
                else:
                    orbit_history["running_best_score"].append(prev_best)
                    orbit_history["running_best_sep_target"].append(
                        orbit_history["running_best_sep_target"][-1]
                    )
                    orbit_history["running_best_sep_control"].append(
                        orbit_history["running_best_sep_control"][-1]
                    )
                    orbit_history["running_best_p_normal_target"].append(
                        orbit_history["running_best_p_normal_target"][-1]
                    )
                    orbit_history["running_best_p_pi_target"].append(
                        orbit_history["running_best_p_pi_target"][-1]
                    )
                    orbit_history["running_best_p_normal_control"].append(
                        orbit_history["running_best_p_normal_control"][-1]
                    )
                    orbit_history["running_best_p_pi_control"].append(
                        orbit_history["running_best_p_pi_control"][-1]
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
                measurement_streams[qp.name] = {"orbit_history": orbit_history}
            finally:
                job.cancel()

    node.namespace["optimization_results"] = optimization_results
    node.namespace["measurement_streams"] = measurement_streams
    node.results["orbit_depth"] = node.parameters.orbit_depth
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
def load_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["orbit_depth"] = node.parameters.orbit_depth
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
def analyse_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Summarise the CMA-ES optimisation outcome per qubit pair."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )

    for qp_name, summary in fit_results.items():
        streams = measurement_streams.get(qp_name, {})
        orbit_hist = streams.get("orbit_history", {})
        if orbit_hist:
            summary["best_separation_target"] = max(orbit_hist["separation_target"])
            summary["best_separation_control"] = max(orbit_hist["separation_control"])
            summary["best_score_average"] = max(orbit_hist["score_average"])

    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    for qp in node.namespace["qubit_pairs"]:
        summary = fit_results.get(qp.name)
        if summary is None:
            continue
        sep_t = summary.get("best_separation_target")
        sep_c = summary.get("best_separation_control")
        score_avg = summary.get("best_score_average")
        if sep_t is not None:
            node.log(
                f"  [{qp.name}] Best orbit separations: "
                f"{qp.qubit_target.name}={sep_t:.4f}, "
                f"{qp.qubit_control.name}={sep_c:.4f}, "
                f"average={score_avg:.4f} "
                f"(at depth {node.parameters.orbit_depth})"
            )

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qp_name, summary in fit_results.items()
    }


# %% {Plot_data}
def _plot_orbit_separation_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot orbit separation for the best candidate of each generation."""
    if not orbit_history or not orbit_history.get("score_average"):
        ax.set_title(f"No orbit data — {pair_name}")
        return

    n_gen = len(orbit_history["score_average"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["separation_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name}",
    )
    ax.plot(
        generations, orbit_history["separation_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name}",
    )
    ax.plot(
        generations, orbit_history["score_average"], "D-",
        color="C2", markersize=5, linewidth=2, label="Average",
    )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Orbit separation (P_normal − P_pi)")
    title = (
        f"Best-candidate orbit separation — {pair_name}" if pair_name
        else "Best-candidate orbit separation"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_survival_variants_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot P_normal and P_pi for the best candidate of each generation."""
    if not orbit_history or not orbit_history.get("p_normal_target"):
        ax.set_title(f"No survival data — {pair_name}")
        return

    n_gen = len(orbit_history["p_normal_target"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["p_normal_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name} normal",
    )
    ax.plot(
        generations, orbit_history["p_pi_target"], "o--",
        color="C0", markersize=4, alpha=0.6, label=f"{qubit_target_name} π-init",
    )
    ax.plot(
        generations, orbit_history["p_normal_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name} normal",
    )
    ax.plot(
        generations, orbit_history["p_pi_control"], "s--",
        color="C1", markersize=4, alpha=0.6, label=f"{qubit_control_name} π-init",
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival probability")
    title = (
        f"Best-candidate survival — {pair_name}" if pair_name
        else "Best-candidate survival"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_individual_scores_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Scatter all candidate scores per generation with running best overlay."""
    if not orbit_history or not orbit_history.get("all_score_average"):
        ax.set_title(f"No individual score data — {pair_name}")
        return

    n_gen = len(orbit_history["all_score_average"])
    for gen_idx in range(n_gen):
        gen_num = gen_idx + 1
        avg_vals = np.asarray(orbit_history["all_score_average"][gen_idx])
        ax.scatter(
            np.full_like(avg_vals, gen_num, dtype=float), avg_vals,
            s=10, alpha=0.3, color="C7", zorder=1,
        )

    generations = np.arange(1, n_gen + 1)
    if orbit_history.get("running_best_score"):
        ax.plot(
            generations, orbit_history["running_best_score"],
            "D-", color="C2", markersize=5, linewidth=2,
            label="Running best (avg)", zorder=3,
        )
    if orbit_history.get("running_best_sep_target"):
        ax.plot(
            generations, orbit_history["running_best_sep_target"],
            "o--", color="C0", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_target_name})", zorder=2,
        )
    if orbit_history.get("running_best_sep_control"):
        ax.plot(
            generations, orbit_history["running_best_sep_control"],
            "s--", color="C1", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_control_name})", zorder=2,
        )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Orbit score")
    title = (
        f"Individual scores & running best — {pair_name}" if pair_name
        else "Individual scores & running best"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_running_best_survival_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    orbit_depth: int,
    pair_name: str = "",
) -> None:
    """Plot the running-best candidate's P_normal and P_pi over generations."""
    if not orbit_history or not orbit_history.get("running_best_p_normal_target"):
        ax.set_title(f"No running-best data — {pair_name}")
        return

    n_gen = len(orbit_history["running_best_p_normal_target"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["running_best_p_normal_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name} normal",
    )
    ax.plot(
        generations, orbit_history["running_best_p_pi_target"], "o--",
        color="C0", markersize=4, alpha=0.6, label=f"{qubit_target_name} π-init",
    )
    ax.plot(
        generations, orbit_history["running_best_p_normal_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name} normal",
    )
    ax.plot(
        generations, orbit_history["running_best_p_pi_control"], "s--",
        color="C1", markersize=4, alpha=0.6, label=f"{qubit_control_name} π-init",
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    final_score = orbit_history.get("running_best_score", [])
    if final_score:
        ax.text(
            0.98, 0.02,
            f"Best score: {final_score[-1]:.4f} (depth={orbit_depth})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival probability")
    title = (
        f"Running-best survival — {pair_name}" if pair_name
        else "Running-best survival"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Generate convergence, separation, and parameter-evolution plots."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    pair_names = list(opt_results.keys())
    n_pairs = max(len(pair_names), 1)
    orbit_depth = node.parameters.orbit_depth

    fig_combined, axes = plt.subplots(
        5, n_pairs, figsize=(9 * n_pairs, 22), squeeze=False,
    )

    for col, pname in enumerate(pair_names):
        streams = measurement_streams.get(pname, {})
        orbit_hist = streams.get("orbit_history", {})

        qp = node.namespace["qubit_pairs"][col]

        _plot_orbit_separation_on_ax(
            axes[0, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )
        _plot_survival_variants_on_ax(
            axes[1, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )
        _plot_running_best_survival_on_ax(
            axes[2, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name,
            orbit_depth, pname,
        )
        plot_score_convergence_on_ax(axes[3, col], opt_results[pname], pname)
        _plot_individual_scores_on_ax(
            axes[4, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )

    fig_combined.tight_layout()

    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    node.results["figures"] = {
        "orbit_and_convergence": fig_combined,
        "parameter_evolution": fig_params,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[CMAESOrbitParameters, Quam]):
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
            sep_t = pair_summary.get("best_separation_target", pair_summary["best_score"])
            sep_c = pair_summary.get("best_separation_control", pair_summary["best_score"])

            qubit_params = [
                (qp.qubit_target, best["amplitude_scale_target"], best["duration_offset_target"], best["freq_detuning_target"], sep_t),
                (qp.qubit_control, best["amplitude_scale_control"], best["duration_offset_control"], best["freq_detuning_control"], sep_c),
            ]

            for qubit, opt_amp_scale, opt_dur_offset, opt_freq_det, separation in qubit_params:
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
                    f"orbit_separation={separation:.4f}"
                )

            node.log(
                f"  {qp.name}: average orbit score = {pair_summary['best_score']:.4f} "
                f"(depth={node.parameters.orbit_depth})"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Persist all results, figures, and parameters to disk."""
    node.save()
