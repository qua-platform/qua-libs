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
from calibration_utils.cmaes.reset_initialization_parameters import ResetInitializationParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from quam_builder.architecture.quantum_dots.operations.voltage_balanced_macros.state_macros import (
    BalancedInitializeMacroWithConditionalDrive,
)


# %% {Node initialisation}
description = """
        OPTIMISE RESET INITIALISATION (CMA-ES) — VISIBILITY
Uses CMA-ES to optimise the BalancedInitializeMacroWithConditionalDrive by tuning:
  - the amplitude scale of the conditional x pulse,
  - the intermediate frequency detuning of that drive,
  - the ramp duration,
  - the hold duration,
  - the barrier gate voltage (in the initialize voltage point),
  - and the detuning-axis voltage (in the initialize voltage point).

The conditional drive macro:
  1. Ramps to −V (initialisation point), measures in PSB, conditionally
     applies an x pulse to flip spin if not in ground state.
  2. Ramps to +V, measures again, conditionally applies x pulse.
  3. Ramps back to 0.

Three experiment types are interleaved per shot:
    1. Reference — reset init, then measure (no additional pi pulse).
    2. Pi on target — reset init, x180 on target qubit, then measure.
    3. Pi on control — reset init, x180 on control qubit, then measure.

The objective is the mean visibility:

    score = ( |mean(pi_target) − mean(ref)| + |mean(pi_control) − mean(ref)| ) / 2

CMA-ES maximises this score.

x_amplitude, if_detuning, ramp_duration, and hold_duration are varied per-
candidate via QUA input streams (update_correction / update_frequency / macro
kwargs).  barrier and detuning are set once per generation by recompiling the
QUA program with the generation-mean values; all candidates in a generation
share those voltage-point values.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.
    - Having calibrated pi pulses (x180) on both qubits.
    - Having BalancedInitializeMacroWithConditionalDrive installed as the
      initialize macro on the relevant quantum dot pairs.

State update:
    - Updates ramp_duration and hold_duration on the macro.
    - Updates the initialize voltage point with the optimal barrier and
      detuning voltages.
    - Shifts the xy channel intermediate_frequency by the optimal detuning.
    - Scales the x180 pulse amplitude by the optimal amplitude factor.
"""

node = QualibrationNode[ResetInitializationParameters, Quam](
    name="02a_optimize_reset_initialize",
    description=description,
    parameters=ResetInitializationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 20
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 10
    node.parameters.population_size = 20
    # node.parameters.simulate = True


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


_AMP_IDX = 0
_IF_IDX = 1
_RAMP_IDX = 2
_HOLD_IDX = 3
_BAR_IDX = 4
_DET_IDX = 5

_PARAM_NAMES = [
    "x_amplitude",
    "if_detuning",
    "ramp_duration",
    "hold_duration",
    "barrier",
    "detuning",
]


def _set_init_point_voltages(
    dot_pair, barrier_val: float, detuning_val: float
) -> None:
    """Update the barrier and detuning components of the 'initialize' voltage point."""
    barrier_id = dot_pair.barrier_gate.id
    detuning_axis = dot_pair.name
    dot_pair.add_point(
        "initialize",
        voltages={barrier_id: barrier_val, detuning_axis: detuning_val},
        duration=16,
        replace_existing_point=True,
    )


def _build_qua_program(
    node, qubit_pair, barrier_val: float, detuning_val: float
):
    """Build a QUA program for a single qubit pair at fixed voltage-point values.

    x_amplitude, if_detuning, ramp_duration, and hold_duration are varied per-
    candidate via input streams.  barrier_val and detuning_val are compile-time
    constants baked into the initialize voltage point; call this once per
    generation when either changes.

    For each candidate and shot, three experiment types are run:
        1. Reference — reset init ramp, then PSB measure (no extra pi pulse).
        2. Pi target — reset init ramp, x180 on target qubit, then measure.
        3. Pi control — reset init ramp, x180 on control qubit, then measure.
    """
    dot_pair = qubit_pair.quantum_dot_pair
    pop_size = node.parameters.population_size
    num_shots = node.parameters.num_shots

    init_macro = qubit_pair.macros.get("initialize")
    if not isinstance(init_macro, BalancedInitializeMacroWithConditionalDrive):
        raise TypeError(
            f"Expected BalancedInitializeMacroWithConditionalDrive on {dot_pair.name}, "
            f"got {type(init_macro).__name__}. Install the macro before running this node."
        )

    _set_init_point_voltages(dot_pair, barrier_val, detuning_val)

    xy_ch = qubit_pair.xy
    base_if = int(xy_ch.intermediate_frequency)

    with program() as qua_prog:
        candidate_idx = declare(int)

        amp_in = declare_input_stream("client", stream_id="x_amplitude", dtype=fixed)
        if_in = declare_input_stream("client", stream_id="if_detuning", dtype=int)
        rd_in = declare_input_stream("client", stream_id="ramp_duration", dtype=int)
        hd_in = declare_input_stream("client", stream_id="hold_duration", dtype=int)

        amp_q = declare(fixed)
        if_q = declare(int)
        rd_q = declare(int)
        hd_q = declare(int)

        state_ref = declare(int)
        state_pi_tgt = declare(int)
        state_pi_ctl = declare(int)
        state_ref_st = declare_output_stream()
        state_pi_tgt_st = declare_output_stream()
        state_pi_ctl_st = declare_output_stream()

        shot = declare(int)

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(amp_in)
                advance_input_stream(if_in)
                advance_input_stream(rd_in)
                advance_input_stream(hd_in)
                assign(amp_q, amp_in)
                assign(if_q, if_in)
                assign(rd_q, rd_in)
                assign(hd_q, hd_in)

                update_frequency(xy_ch.id, base_if + if_q)

                with for_(shot, 0, shot < num_shots, shot + 1):
                    # --- Type 1: reference (no extra pi pulse) ---
                    qubit_pair.initialize(
                        ramp_duration=rd_q,
                        hold_duration=hd_q,
                        amplitude=amp_q,
                        conditional_drive=True,
                    )
                    assign(state_ref, Cast.to_int(dot_pair.measure()))
                    save(state_ref, state_ref_st)
                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

                    # --- Type 2: pi pulse on target qubit ---
                    qubit_pair.initialize(
                        ramp_duration=rd_q,
                        hold_duration=hd_q,
                        amplitude=amp_q,
                        conditional_drive=True,
                    )
                    align()
                    qubit_pair.qubit_target.x180()
                    align()
                    assign(state_pi_tgt, Cast.to_int(dot_pair.measure()))
                    save(state_pi_tgt, state_pi_tgt_st)
                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

                    # --- Type 3: pi pulse on control qubit ---
                    qubit_pair.initialize(
                        ramp_duration=rd_q,
                        hold_duration=hd_q,
                        amplitude=amp_q,
                        conditional_drive=True,
                    )
                    align()
                    qubit_pair.qubit_control.x180()
                    align()
                    assign(state_pi_ctl, Cast.to_int(dot_pair.measure()))
                    save(state_pi_ctl, state_pi_ctl_st)
                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

        update_frequency(xy_ch.id, base_if + if_q)

        with stream_processing():
            state_ref_st.buffer(num_shots).buffer(pop_size).save_all("state_ref")
            state_pi_tgt_st.buffer(num_shots).buffer(pop_size).save_all("state_pi_target")
            state_pi_ctl_st.buffer(num_shots).buffer(pop_size).save_all("state_pi_control")

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Validate parameters, resolve qubit pairs, and build a QUA program
    for simulation (first pair only, at the initial barrier value).
    """
    if node.parameters.num_shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {node.parameters.num_shots}")
    if node.parameters.population_size < 1:
        raise ValueError(
            f"population_size must be >= 1, got {node.parameters.population_size}"
        )

    qubit_pairs = _resolve_qubit_pairs(node)
    if not qubit_pairs:
        raise ValueError(
            "No qubit pairs resolved — check qubit_pairs parameter or machine config."
        )
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = dot_pair_objects

    node.namespace["qua_program"] = _build_qua_program(
        node,
        qubit_pairs[0],
        node.parameters.barrier_initial,
        node.parameters.detuning_initial,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[ResetInitializationParameters, Quam]):
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


def _compute_visibility(
    ref_avg: np.ndarray,
    pi_target_avg: np.ndarray,
    pi_control_avg: np.ndarray,
) -> np.ndarray:
    """Compute visibility scores from shot-averaged experiment data."""
    ref = ref_avg.astype(np.float64)
    pi_tgt = pi_target_avg.astype(np.float64)
    pi_ctl = pi_control_avg.astype(np.float64)
    return (np.abs(pi_tgt - ref) + np.abs(pi_ctl - ref)) / 2.0


def _quantize_duration(val: float, min_val: int = 16) -> int:
    """Quantize a continuous duration to a multiple of 4 ns, clamped."""
    return max(min_val, int(round(val / 4.0)) * 4)


# %% {Run_CMA-ES_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_cmaes_loop(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    x_amplitude, if_detuning, ramp_duration, and hold_duration are varied per-
    candidate via input streams.  barrier and detuning are updated once per
    generation: when the generation-mean value of either shifts by more than
    0.1 mV, the QUA program is recompiled and the job is restarted.
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    pop_size = node.parameters.population_size

    # Normalise all parameters to [0, 1] so that sigma0 is meaningful for
    # every dimension.  The physical scales span orders of magnitude
    # (IF detuning ~20 MHz vs barrier ~0.2 V vs amplitude ~1.9); without
    # normalisation CMA-ES barely explores the larger-range dimensions.
    # We denormalise inside evaluate_candidates and convert opt_result
    # back to physical units after optimisation.
    lo = np.array([
        node.parameters.x_amplitude_min,
        float(node.parameters.if_detuning_min),
        float(node.parameters.ramp_duration_min),
        float(node.parameters.hold_duration_min),
        node.parameters.barrier_min,
        node.parameters.detuning_min,
    ])
    hi = np.array([
        node.parameters.x_amplitude_max,
        float(node.parameters.if_detuning_max),
        float(node.parameters.ramp_duration_max),
        float(node.parameters.hold_duration_max),
        node.parameters.barrier_max,
        node.parameters.detuning_max,
    ])
    param_range = hi - lo

    x0_phys = np.array([
        node.parameters.x_amplitude_initial,
        float(node.parameters.if_detuning_initial),
        float(node.parameters.ramp_duration_initial),
        float(node.parameters.hold_duration_initial),
        node.parameters.barrier_initial,
        node.parameters.detuning_initial,
    ])
    x0_norm = (x0_phys - lo) / param_range
    bounds_norm = [(0.0, 1.0)] * len(_PARAM_NAMES)

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES reset-init optimisation for {qp.name}...")

        pair_streams = {"ref": [], "pi_target": [], "pi_control": []}

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            current_barrier = node.parameters.barrier_initial
            current_detuning = node.parameters.detuning_initial
            current_job = qm.execute(
                _build_qua_program(node, qp, current_barrier, current_detuning)
            )
            ref_handle = current_job.result_handles.get("state_ref")
            pi_tgt_handle = current_job.result_handles.get("state_pi_target")
            pi_ctl_handle = current_job.result_handles.get("state_pi_control")
            generation_counter = 0

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                nonlocal current_job, ref_handle, pi_tgt_handle, pi_ctl_handle
                nonlocal current_barrier, current_detuning, generation_counter

                candidates = lo + candidates_norm * param_range

                gen_barrier = float(np.mean(candidates[:, _BAR_IDX]))
                gen_detuning = float(np.mean(candidates[:, _DET_IDX]))

                barrier_changed = abs(gen_barrier - current_barrier) > 1e-4
                detuning_changed = abs(gen_detuning - current_detuning) > 1e-4

                if barrier_changed or detuning_changed:
                    node.log(
                        f"    Recompiling for barrier={gen_barrier:.6g} V, "
                        f"detuning={gen_detuning:.6g} V"
                    )
                    current_job.cancel()
                    current_barrier = gen_barrier
                    current_detuning = gen_detuning
                    generation_counter = 0
                    current_job = qm.execute(
                        _build_qua_program(
                            node, qp, current_barrier, current_detuning
                        )
                    )
                    ref_handle = current_job.result_handles.get("state_ref")
                    pi_tgt_handle = current_job.result_handles.get("state_pi_target")
                    pi_ctl_handle = current_job.result_handles.get("state_pi_control")

                for val in candidates[:, _AMP_IDX].tolist():
                    current_job.push_to_input_stream("x_amplitude", float(val))
                for val in candidates[:, _IF_IDX].tolist():
                    current_job.push_to_input_stream("if_detuning", int(val))
                for val in candidates[:, _RAMP_IDX]:
                    current_job.push_to_input_stream(
                        "ramp_duration", _quantize_duration(val)
                    )
                for val in candidates[:, _HOLD_IDX]:
                    current_job.push_to_input_stream(
                        "hold_duration", _quantize_duration(val)
                    )

                target_count = generation_counter + 1
                while ref_handle.count_so_far() < target_count:
                    _time.sleep(0.005)
                while pi_tgt_handle.count_so_far() < target_count:
                    _time.sleep(0.005)
                while pi_ctl_handle.count_so_far() < target_count:
                    _time.sleep(0.005)

                raw_ref = np.asarray(
                    ref_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                ).mean(axis=-1)
                raw_pi_tgt = np.asarray(
                    pi_tgt_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                ).mean(axis=-1)
                raw_pi_ctl = np.asarray(
                    pi_ctl_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                ).mean(axis=-1)
                generation_counter += 1

                pair_streams["ref"].append(raw_ref.copy())
                pair_streams["pi_target"].append(raw_pi_tgt.copy())
                pair_streams["pi_control"].append(raw_pi_ctl.copy())

                return _compute_visibility(raw_ref, raw_pi_tgt, raw_pi_ctl)

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
                measurement_streams[qp.name] = pair_streams
            finally:
                current_job.cancel()

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
def load_data(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)
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
def analyse_data(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Summarise the CMA-ES optimisation outcome per qubit pair."""
    opt_results = node.namespace["optimization_results"]

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )
    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qp_name, summary in fit_results.items()
    }


# %% {Plot_data}
def _plot_measurement_streams_on_ax(
    ax: plt.Axes,
    streams: dict,
    pair_name: str = "",
) -> None:
    """Plot ref / pi-target / pi-control mean±std vs generation."""
    labels_colors = [
        ("ref", "Reference (no pi)", "C0"),
        ("pi_target", "Pi target", "C1"),
        ("pi_control", "Pi control", "C2"),
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

    ax.set_xlabel("Generation")
    ax.set_ylabel("Measurement probability")
    title = (
        f"Measurement streams — {pair_name}" if pair_name
        else "Measurement streams"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Generate convergence and parameter-evolution plots (per qubit pair)."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    pair_names = list(opt_results.keys())
    n_pairs = max(len(pair_names), 1)

    fig_combined, axes = plt.subplots(
        2, n_pairs, figsize=(7 * n_pairs, 8), squeeze=False,
    )

    for col, pair_name in enumerate(pair_names):
        streams = measurement_streams.get(pair_name)
        if streams:
            _plot_measurement_streams_on_ax(axes[0, col], streams, pair_name)
        else:
            axes[0, col].set_title(f"No measurement data — {pair_name}")

        plot_score_convergence_on_ax(axes[1, col], opt_results[pair_name], pair_name)

    fig_combined.tight_layout()

    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    node.results["figures"] = {
        "streams_and_convergence": fig_combined,
        "parameter_evolution": fig_params,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[ResetInitializationParameters, Quam]):
    """Apply the optimal parameters to the BalancedInitializeMacroWithConditionalDrive."""
    fit_results = node.results["fit_results"]

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            pair_summary = fit_results.get(qp.name)
            if pair_summary is None:
                continue
            if not pair_summary["success"]:
                node.log(f"  {qp.name}: optimisation did not succeed — skipping.")
                continue

            best = pair_summary["best_params"]
            dot_pair = qp.quantum_dot_pair
            xy_ch = qp.qubit_target.xy

            opt_rd = _quantize_duration(best["ramp_duration"])
            opt_hd = _quantize_duration(best["hold_duration"])

            init_macro = dot_pair.macros.get("initialize")
            if isinstance(init_macro, BalancedInitializeMacroWithConditionalDrive):
                init_macro.ramp_duration = opt_rd
                init_macro.hold_duration = opt_hd

            _set_init_point_voltages(
                dot_pair, best["barrier"], best["detuning"]
            )

            xy_ch.intermediate_frequency = int(
                xy_ch.intermediate_frequency + best["if_detuning"]
            )

            x180_pulse = xy_ch.operations.get("x180")
            if x180_pulse is not None and hasattr(x180_pulse, "amplitude"):
                x180_pulse.amplitude *= best["x_amplitude"]

            node.log(
                f"  {qp.name}: reset-init updated — "
                f"x_amplitude_scale={best['x_amplitude']:.4f}, "
                f"if_detuning={best['if_detuning']:.0f} Hz, "
                f"ramp={opt_rd} ns, hold={opt_hd} ns, "
                f"barrier={best['barrier']:.6g} V, "
                f"detuning={best['detuning']:.6g} V, "
                f"visibility={pair_summary['best_score']:.4f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[ResetInitializationParameters, Quam]):
    node.save()