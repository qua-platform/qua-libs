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
from calibration_utils.cmaes.heralded_initialization_parameters import HeraldedInitializationParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot
from quam_builder.architecture.quantum_dots.operations.voltage_balanced_macros.state_macros import (
    BalancedHeraldedInitializeMacro,
)


# %% {Node initialisation}
description = """
        OPTIMISE HERALDED INITIALISATION (CMA-ES) — VISIBILITY
Uses CMA-ES to optimise the BalancedHeraldedInitializeMacro by tuning:
  - the ramp duration of the initialization voltage ramp,
  - the hold duration at the initialization voltage point,
  - the ramp duration of the internal heralded PSB measurement,
  - and the buffer duration before the internal heralded PSB readout.

The heralded initialisation macro loops until the PSB measurement returns
the target state (or max_loops is reached), applying a conditional x pulse
on the control qubit when the state is not the target.

Three experiment types are interleaved per shot:
    1. Reference — heralded init, then measure (no additional pi pulse).
    2. Pi on target — heralded init, x180 on target qubit, then measure.
    3. Pi on control — heralded init, x180 on control qubit, then measure.

The objective is the mean visibility:

    score = ( |mean(pi_target) − mean(ref)| + |mean(pi_control) − mean(ref)| ) / 2

CMA-ES maximises this score.

All four parameters are varied per-candidate via QUA input streams.
The QUA program is compiled once per qubit pair.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.
    - Having calibrated pi pulses (x180) on both qubits.
    - Having BalancedHeraldedInitializeMacro installed as the initialize
      macro on the relevant quantum dot pairs.

State update:
    - Updates ramp_duration and hold_duration on the BalancedHeraldedInitializeMacro.
    - Updates ramp_duration and buffer_duration on the measure macro.
"""

node = QualibrationNode[HeraldedInitializationParameters, Quam](
    name="02b_optimize_heralded_initialize",
    description=description,
    parameters=HeraldedInitializationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 20
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 10
    node.parameters.population_size = 20
    # node.parameters.simulate = True


node.machine = Quam.load()


def _resolve_qubit_pairs(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


_INIT_RAMP_IDX = 0
_INIT_HOLD_IDX = 1
_MEAS_RAMP_IDX = 2
_MEAS_BUF_IDX = 3

_PARAM_NAMES = [
    "init_ramp_duration",
    "init_hold_duration",
    "meas_ramp_duration",
    "meas_buffer_duration",
]


def _build_qua_program(node, qubit_pair):
    """Build a QUA program for a single qubit pair.

    All four timing parameters are varied per-candidate via input streams.
    The QUA program is compiled once per qubit pair and reused across all
    generations.

    For each candidate and shot, three experiment types are run:
        1. Reference — heralded init, then PSB measure (no extra pi pulse).
        2. Pi target — heralded init, x180 on target qubit, then measure.
        3. Pi control — heralded init, x180 on control qubit, then measure.
    """
    dot_pair = qubit_pair.quantum_dot_pair
    pop_size = node.parameters.population_size
    num_shots = node.parameters.num_shots
    qubit_name = qubit_pair.qubit_control.name

    init_macro = qubit_pair.quantum_dot_pair.macros.get("initialize")
    if not isinstance(init_macro, BalancedHeraldedInitializeMacro):
        raise TypeError(
            f"Expected BalancedHeraldedInitializeMacro on {dot_pair.name}, "
            f"got {type(init_macro).__name__}. Install the macro before running this node."
        )

    with program() as qua_prog:
        candidate_idx = declare(int)

        ird_in = declare_input_stream("client", stream_id="init_ramp_duration", dtype=int)
        ihd_in = declare_input_stream("client", stream_id="init_hold_duration", dtype=int)
        mrd_in = declare_input_stream("client", stream_id="meas_ramp_duration", dtype=int)
        mbd_in = declare_input_stream("client", stream_id="meas_buffer_duration", dtype=int)

        ird_q = declare(int)
        ihd_q = declare(int)
        mrd_q = declare(int)
        mbd_q = declare(int)

        state_ref = declare(int)
        state_pi_tgt = declare(int)
        state_pi_ctl = declare(int)
        state_ref_st = declare_output_stream()
        state_pi_tgt_st = declare_output_stream()
        state_pi_ctl_st = declare_output_stream()

        shot = declare(int)

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(ird_in)
                advance_input_stream(ihd_in)
                advance_input_stream(mrd_in)
                advance_input_stream(mbd_in)
                assign(ird_q, ird_in)
                assign(ihd_q, ihd_in)
                assign(mrd_q, mrd_in)
                assign(mbd_q, mbd_in)

                with for_(shot, 0, shot < num_shots, shot + 1):
                    # --- Type 1: reference (no extra pi pulse) ---
                    qubit_pair.initialize(
                        ramp_duration=ird_q,
                        hold_duration=ihd_q,
                        meas_ramp_duration=mrd_q,
                        meas_buffer_duration=mbd_q,
                        qubit_name=qubit_name,
                        conditional_drive=True,
                    )
                    assign(state_ref, Cast.to_int(dot_pair.measure()))
                    save(state_ref, state_ref_st)
                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

                    # --- Type 2: pi pulse on target qubit ---
                    qubit_pair.initialize(
                        ramp_duration=ird_q,
                        hold_duration=ihd_q,
                        meas_ramp_duration=mrd_q,
                        meas_buffer_duration=mbd_q,
                        qubit_name=qubit_name,
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
                        ramp_duration=ird_q,
                        hold_duration=ihd_q,
                        meas_ramp_duration=mrd_q,
                        meas_buffer_duration=mbd_q,
                        qubit_name=qubit_name,
                        conditional_drive=True,
                    )
                    align()
                    qubit_pair.qubit_control.x180()
                    align()
                    assign(state_pi_ctl, Cast.to_int(dot_pair.measure()))
                    save(state_pi_ctl, state_pi_ctl_st)
                    align()
                    dot_pair.voltage_sequence.ramp_to_zero()

        with stream_processing():
            (
                state_ref_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("state_ref")
            )
            (
                state_pi_tgt_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("state_pi_target")
            )
            (
                state_pi_ctl_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("state_pi_control")
            )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    """Validate parameters, resolve qubit pairs, and build a QUA program
    for simulation (first pair only).
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

    node.namespace["qua_program"] = _build_qua_program(node, qubit_pairs[0])


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
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
def run_cmaes_loop(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    All four timing parameters are varied per-candidate via input streams.
    The QUA program is compiled once per qubit pair and reused for all
    generations.
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    pop_size = node.parameters.population_size

    lo_phys = np.array([
        float(node.parameters.init_ramp_duration_min),
        float(node.parameters.init_hold_duration_min),
        float(node.parameters.meas_ramp_duration_min),
        float(node.parameters.meas_buffer_duration_min),
    ])
    hi_phys = np.array([
        float(node.parameters.init_ramp_duration_max),
        float(node.parameters.init_hold_duration_max),
        float(node.parameters.meas_ramp_duration_max),
        float(node.parameters.meas_buffer_duration_max),
    ])

    if node.parameters.log_norm:
        lo = np.log(lo_phys)
        hi = np.log(hi_phys)
        def _to_phys(x_norm): return np.exp(lo + x_norm * (hi - lo))
        def _to_norm(x_phys): return (np.log(x_phys) - lo) / (hi - lo)
    else:
        lo = lo_phys
        hi = hi_phys
        def _to_phys(x_norm): return lo + x_norm * (hi - lo)
        def _to_norm(x_phys): return (x_phys - lo) / (hi - lo)

    x0_norm = np.full(len(_PARAM_NAMES), 0.5)
    bounds_norm = [(0.0, 1.0)] * len(_PARAM_NAMES)

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES heralded-init optimisation for {qp.name}...")

        pair_streams = {"ref": [], "pi_target": [], "pi_control": []}

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            current_job = qm.execute(_build_qua_program(node, qp))
            ref_handle = current_job.result_handles.get("state_ref")
            pi_tgt_handle = current_job.result_handles.get("state_pi_target")
            pi_ctl_handle = current_job.result_handles.get("state_pi_control")
            generation_counter = 0

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                nonlocal generation_counter

                candidates = _to_phys(candidates_norm)

                for val in candidates[:, _INIT_RAMP_IDX]:
                    current_job.push_to_input_stream(
                        "init_ramp_duration", _quantize_duration(val)
                    )
                for val in candidates[:, _INIT_HOLD_IDX]:
                    current_job.push_to_input_stream(
                        "init_hold_duration", _quantize_duration(val)
                    )
                for val in candidates[:, _MEAS_RAMP_IDX]:
                    current_job.push_to_input_stream(
                        "meas_ramp_duration", _quantize_duration(val)
                    )
                for val in candidates[:, _MEAS_BUF_IDX]:
                    current_job.push_to_input_stream(
                        "meas_buffer_duration", _quantize_duration(val)
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
                )
                raw_pi_tgt = np.asarray(
                    pi_tgt_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
                raw_pi_ctl = np.asarray(
                    pi_ctl_handle.fetch(generation_counter, flat_struct=True),
                    dtype=np.float64,
                )
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
                opt_result.best_params = _to_phys(opt_result.best_params)
                opt_result.param_history = [
                    _to_phys(h) for h in opt_result.param_history
                ]
                opt_result.all_candidates = [
                    _to_phys(c) for c in opt_result.all_candidates
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
def load_data(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
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
def analyse_data(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
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
def plot_data(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
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
def update_state(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    """Apply the optimal parameters to the BalancedHeraldedInitializeMacro and measure macro."""
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

            opt_ird = _quantize_duration(best["init_ramp_duration"])
            opt_ihd = _quantize_duration(best["init_hold_duration"])
            opt_mrd = _quantize_duration(best["meas_ramp_duration"])
            opt_mbd = _quantize_duration(best["meas_buffer_duration"])

            init_macro = dot_pair.macros.get("initialize")
            if isinstance(init_macro, BalancedHeraldedInitializeMacro):
                init_macro.ramp_duration = opt_ird
                init_macro.hold_duration = opt_ihd

            from quam_builder.architecture.quantum_dots.operations.names import TwoQubitMacroName
            from quam_builder.architecture.quantum_dots.operations.voltage_balanced_macros.state_macros import (
                BalancedMeasurePSBPairMacro,
            )
            measure_macro = dot_pair.macros.get(TwoQubitMacroName.MEASURE)
            if isinstance(measure_macro, BalancedMeasurePSBPairMacro):
                measure_macro.ramp_duration = opt_mrd
                measure_macro.buffer_duration = opt_mbd

            node.log(
                f"  {qp.name}: heralded-init updated — "
                f"init_ramp={opt_ird} ns, init_hold={opt_ihd} ns, "
                f"meas_ramp={opt_mrd} ns, meas_buffer={opt_mbd} ns, "
                f"visibility={pair_summary['best_score']:.4f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[HeraldedInitializationParameters, Quam]):
    node.save()