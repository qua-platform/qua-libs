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
from calibration_utils.cmaes.initialization_parameters import InitializationParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from quam_builder.architecture.quantum_dots.operations.voltage_balanced_macros.state_macros import (
    BalancedInitializeMacro,
)


# %% {Node initialisation}
description = """
        OPTIMISE INITIALISATION (CMA-ES) — VISIBILITY
Uses CMA-ES to optimise a single balanced initialisation ramp by tuning
the detuning and barrier voltages, ramp duration, and hold duration.

The waveform is anti-symmetric (balanced):

    0 → −V → hold → +V → hold → 0

Three experiment types are interleaved per shot:
    1. Reference — init ramp then measure (no pi pulse).
    2. Pi on target — init ramp, pi pulse on target qubit, then measure.
    3. Pi on control — init ramp, pi pulse on control qubit, then measure.

The objective is the mean visibility:

    score = ( |mean(pi_target) − mean(ref)| + |mean(pi_control) − mean(ref)| ) / 2

which ranges from 0 (no separation) to 1 (perfect).  CMA-ES maximises
this score.

The QUA program is compiled once per qubit pair and uses input streams
so that each new generation only requires pushing fresh parameter values.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.
    - Having calibrated pi pulses (x180) on both qubits.

State update:
    - Replaces the initialise macro on each qubit pair with a
      ``BalancedInitializeMacro`` configured with the optimal
      ramp duration, hold duration, and voltage point.
"""

node = QualibrationNode[InitializationParameters, Quam](
    name="02_optimize_initialization",
    description=description,
    parameters=InitializationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[InitializationParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 20
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 10
    node.parameters.population_size = 20
    # node.parameters.simulate = True


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[InitializationParameters, Quam]):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


_PARAM_NAMES = [
    "detuning",
    "barrier",
    "ramp_duration",
    "hold_duration",
]


def _build_qua_program(node, qubit_pair):
    """Build a QUA program for a single qubit pair.

    The program uses ``infinite_loop_`` so it stays alive across CMA-ES
    generations.  ``advance_input_stream`` blocks until Python pushes new
    candidate parameters, so the OPX idles between generations with zero
    recompilation overhead.  Python cancels the job when optimisation is
    complete.

    For each candidate and shot, three experiment types are run:

        1. Reference — balanced init ramp, then PSB measure (no pi pulse).
        2. Pi target — balanced init ramp, x180 on target qubit, then measure.
        3. Pi control — balanced init ramp, x180 on control qubit, then measure.

    The balanced ramp is inlined: 0 → −V → hold → +V → hold → 0.
    All dynamic values are pre-assigned to simple QUA variables to avoid
    nested expressions that can trip the OPX compiler.
    """
    dot_pair = qubit_pair.quantum_dot_pair
    pop_size = node.parameters.population_size
    num_shots = node.parameters.num_shots

    detuning_axis = dot_pair.name
    barrier_id = dot_pair.barrier_gate.id

    vs = dot_pair.voltage_sequence
    vs.limit_play_commands = True
    gates = list(vs.gate_set.channels.keys())

    with program() as qua_prog:
        candidate_idx = declare(int)

        det_in = declare_input_stream("client", stream_id="detuning", dtype=fixed)
        bar_in = declare_input_stream("client", stream_id="barrier", dtype=fixed)
        rd_in = declare_input_stream("client", stream_id="ramp_duration", dtype=int)
        hold_in = declare_input_stream("client", stream_id="hold_duration", dtype=int)

        det_q = declare(fixed)
        bar_q = declare(fixed)
        neg_det_q = declare(fixed)
        neg_bar_q = declare(fixed)
        rd = declare(int)
        double_rd = declare(int)
        hold_dur = declare(int)

        state_ref = declare(int)
        state_pi_tgt = declare(int)
        state_pi_ctl = declare(int)
        state_ref_st = declare_output_stream()
        state_pi_tgt_st = declare_output_stream()
        state_pi_ctl_st = declare_output_stream()

        shot = declare(int)

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(det_in)
                advance_input_stream(bar_in)
                advance_input_stream(rd_in)
                advance_input_stream(hold_in)
                assign(det_q, det_in)
                assign(bar_q, bar_in)
                assign(neg_det_q, -det_in)
                assign(neg_bar_q, -bar_in)
                assign(rd, rd_in)
                assign(double_rd, rd_in << 1)
                assign(hold_dur, hold_in)

                with for_(shot, 0, shot < num_shots, shot + 1):
                    # --- Type 1: reference (no pi pulse) ---
                    align(*gates)
                    vs.ramp_to_voltages({detuning_axis: neg_det_q, barrier_id: neg_bar_q}, duration=hold_dur, ramp_duration=rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: det_q, barrier_id: bar_q}, duration=hold_dur, ramp_duration=double_rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: 0.0, barrier_id: 0.0}, duration=16, ramp_duration=rd, ensure_align=False)
                    align()
                    assign(state_ref, Cast.to_int(dot_pair.measure()))
                    save(state_ref, state_ref_st)

                    # --- Type 2: pi pulse on target qubit ---
                    align(*gates)
                    vs.ramp_to_voltages({detuning_axis: neg_det_q, barrier_id: neg_bar_q}, duration=hold_dur, ramp_duration=rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: det_q, barrier_id: bar_q}, duration=hold_dur, ramp_duration=double_rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: 0.0, barrier_id: 0.0}, duration=16, ramp_duration=rd, ensure_align=False)
                    align()
                    qubit_pair.qubit_target.x180()
                    align()
                    assign(state_pi_tgt, Cast.to_int(dot_pair.measure()))
                    save(state_pi_tgt, state_pi_tgt_st)

                    # --- Type 3: pi pulse on control qubit ---
                    align(*gates)
                    vs.ramp_to_voltages({detuning_axis: neg_det_q, barrier_id: neg_bar_q}, duration=hold_dur, ramp_duration=rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: det_q, barrier_id: bar_q}, duration=hold_dur, ramp_duration=double_rd, ensure_align=False)
                    vs.ramp_to_voltages({detuning_axis: 0.0, barrier_id: 0.0}, duration=16, ramp_duration=rd, ensure_align=False)
                    align()
                    qubit_pair.qubit_control.x180()
                    align()
                    assign(state_pi_ctl, Cast.to_int(dot_pair.measure()))
                    save(state_pi_ctl, state_pi_ctl_st)

        with stream_processing():
            state_ref_st.buffer(num_shots).buffer(pop_size).save_all("state_ref")
            state_pi_tgt_st.buffer(num_shots).buffer(pop_size).save_all("state_pi_target")
            state_pi_ctl_st.buffer(num_shots).buffer(pop_size).save_all("state_pi_control")

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[InitializationParameters, Quam]):
    """Validate parameters, resolve qubit pairs, and build a QUA program
    for simulation (first pair only).

    The per-pair QUA programs used during optimisation are built inside
    ``run_cmaes_loop``.
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

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["qua_program"] = _build_qua_program(node, qubit_pairs[0])


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[InitializationParameters, Quam]):
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
    """Compute visibility scores from shot-averaged experiment data.

    Parameters
    ----------
    ref_avg, pi_target_avg, pi_control_avg : np.ndarray
        Arrays of shape ``(pop_size,)`` containing the mean state
        probability for each experiment type, already averaged over
        shots by stream processing.

    Returns
    -------
    np.ndarray
        Visibility score per candidate, shape ``(pop_size,)``.
        ``(|mean(pi_target) - mean(ref)| + |mean(pi_control) - mean(ref)|) / 2``
    """
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
def run_cmaes_loop(node: QualibrationNode[InitializationParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    A dedicated QUA program is compiled and executed per pair so that only
    the pair under test is driven.  Within each pair's run the program
    stays alive via ``infinite_loop_``; each CMA-ES generation pushes new
    candidate parameters via ``push_to_input_stream``.  The job is
    cancelled once the optimisation finishes (or on error).
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    pop_size = node.parameters.population_size

    # Normalise all parameters to [0, 1] so sigma0 is meaningful for every
    # dimension.  Without normalisation, sigma0=0.01 explores 5 % of the
    # voltage ranges (fine) but only 0.0006 % of the 1800 ns duration ranges
    # (CMA-ES would never reach usable ramp/hold values in the first
    # generations).  Denormalize inside evaluate_candidates and convert
    # opt_result back to physical units after optimisation.
    lo = np.array([
        node.parameters.detuning_min,
        node.parameters.barrier_min,
        float(node.parameters.ramp_duration_min),
        float(node.parameters.hold_duration_min),
    ])
    hi = np.array([
        node.parameters.detuning_max,
        node.parameters.barrier_max,
        float(node.parameters.ramp_duration_max),
        float(node.parameters.hold_duration_max),
    ])
    param_range = hi - lo

    x0_phys = np.array([
        node.parameters.detuning_initial,
        node.parameters.barrier_initial,
        float(node.parameters.ramp_duration_initial),
        float(node.parameters.hold_duration_initial),
    ])
    x0_norm = np.clip((x0_phys - lo) / param_range, 0.01, 0.99)
    bounds_norm = [(0.0, 1.0)] * len(_PARAM_NAMES)

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES initialisation optimisation for {qp.name}...")

        qua_prog = _build_qua_program(node, qp)
        pair_streams = {"ref": [], "pi_target": [], "pi_control": []}

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            result_handles = job.result_handles
            generation_counter = 0

            ref_handle = result_handles.get("state_ref")
            pi_tgt_handle = result_handles.get("state_pi_target")
            pi_ctl_handle = result_handles.get("state_pi_control")

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                """Push one generation (normalised) and fetch visibility scores."""
                nonlocal generation_counter

                candidates_phys = lo + candidates_norm * param_range
                for val in candidates_phys[:, 0].tolist():
                    job.push_to_input_stream("detuning", val)
                for val in candidates_phys[:, 1].tolist():
                    job.push_to_input_stream("barrier", val)
                for val in candidates_phys[:, 2]:
                    job.push_to_input_stream(
                        "ramp_duration", _quantize_duration(val)
                    )
                for val in candidates_phys[:, 3]:
                    job.push_to_input_stream(
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

                return _compute_visibility(
                    raw_ref, raw_pi_tgt, raw_pi_ctl,
                )

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
                # Convert normalised parameter vectors back to physical units so
                # that update_state and load_data work with interpretable values.
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
                job.cancel()

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
def load_data(node: QualibrationNode[InitializationParameters, Quam]):
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
def analyse_data(node: QualibrationNode[InitializationParameters, Quam]):
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
def plot_data(node: QualibrationNode[InitializationParameters, Quam]):
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
def update_state(node: QualibrationNode[InitializationParameters, Quam]):
    """Install a BalancedInitializeMacro with optimal parameters."""
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

            opt_rd = _quantize_duration(best["ramp_duration"])
            opt_hold = _quantize_duration(best["hold_duration"])

            init_macro = dot_pair.macros.get("initialize")
            if isinstance(init_macro, BalancedInitializeMacro):
                init_macro.ramp_duration = opt_rd
                init_macro.hold_duration = opt_hold
                init_macro.point = "initialize"
            else:
                new_macro = BalancedInitializeMacro(
                    ramp_duration=opt_rd,
                    hold_duration=opt_hold,
                    point="initialize",
                )
                set_macro = getattr(dot_pair, "set_macro", None)
                if callable(set_macro):
                    set_macro("initialize", new_macro)
                else:
                    dot_pair.macros["initialize"] = new_macro
                    new_macro.parent = dot_pair

            detuning_axis = dot_pair.name
            barrier_id = dot_pair.barrier_gate.id

            dot_pair.add_point(
                "initialize",
                voltages={
                    detuning_axis: best["detuning"],
                    barrier_id: best["barrier"],
                },
                duration=16,
                replace_existing_point=True,
            )

            node.log(
                f"  {qp.name}: balanced init installed — "
                f"detuning={best['detuning']:.6g} V, "
                f"barrier={best['barrier']:.6g} V, "
                f"ramp={opt_rd} ns, hold={opt_hold} ns, "
                f"visibility={pair_summary['best_score']:.4f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[InitializationParameters, Quam]):
    node.save()
