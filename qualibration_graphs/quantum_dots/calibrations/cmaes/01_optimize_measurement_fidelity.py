# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam
from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName

from calibration_utils.cmaes import (
    Parameters,
    OptimizationResult,
    run_cmaes_optimization,
    analyse_optimization,
    log_optimization_results,
    plot_convergence,
    plot_parameter_evolution,
)
from calibration_utils.iq_sweep.analysis import (
    _pca_projector_np,
    _vmap_em_two_gaussians,
    _two_gaussian_fidelity_visibility,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
        OPTIMISE MEASUREMENT FIDELITY (CMA-ES)
Uses CMA-ES to jointly optimise the measurement detuning and initialisation
ramp duration so as to maximise the readout fidelity.

Each generation of CMA-ES proposes a batch of candidate (detuning,
ramp_duration) pairs.  For every candidate the QUA program initialises the
qubit, ramps to the candidate detuning with the candidate ramp duration, and
performs a dispersive readout for ``num_shots`` repetitions.  The shot-by-shot
I/Q data is projected onto the axis of maximum variance (PCA) and a
two-component Gaussian mixture is fitted analytically to extract the readout
fidelity.

The QUA program is compiled once and uses input streams so that each new
generation only requires pushing fresh parameter values — no recompilation.

Prerequisites:
    - Having initialised the Quam.
    - Having calibrated the PSB measurement point (06a-06c).
    - Having the balanced measurement macro configured with a valid threshold.

State update:
    - The measure voltage point detuning and the initialisation ramp duration
      on each qubit pair.
"""

node = QualibrationNode[Parameters, Quam](
    name="01_optimize_measurement_fidelity",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 5
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 3
    node.parameters.population_size = 2

node.machine = Quam.load()

def _resolve_qubit_pairs(node: QualibrationNode[Parameters, Quam]):
    """Resolve qubit pairs from parameters or default to all machine pairs."""
    if node.parameters.qubit_pairs not in (None, ""):
        return [node.machine.qubit_pairs[name] for name in node.parameters.qubit_pairs]
    return list(node.machine.qubit_pairs.values())


def _build_qua_program(node, qubit_pair):
    """Build a QUA program for a single qubit pair.

    The program uses ``infinite_loop_`` so it stays alive across CMA-ES
    generations.  ``advance_input_stream`` blocks until Python pushes new
    candidate parameters, so the OPX idles between generations with zero
    recompilation overhead.  Python cancels the job when optimisation is
    complete.
    """
    dot_pair = qubit_pair.quantum_dot_pair
    pop_size = node.parameters.population_size
    num_shots = node.parameters.num_shots

    with program() as qua_prog:
        candidate_idx = declare(int)

        detuning_in = declare_input_stream("client", stream_id="detuning", dtype=fixed)
        ramp_dur_in = declare_input_stream("client", stream_id="ramp_duration", dtype=int)
        barrier_in = declare_input_stream("client", stream_id="barrier_voltage", dtype=fixed)

        detuning = declare(fixed)
        ramp_dur = declare(int)
        barrier_voltage = declare(fixed)

        I_var = declare(fixed)
        Q_var = declare(fixed)
        I_st = declare_output_stream()
        Q_st = declare_output_stream()

        shot = declare(int)

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(detuning_in)
                advance_input_stream(ramp_dur_in)
                advance_input_stream(barrier_in)
                assign(detuning, detuning_in)
                assign(ramp_dur, ramp_dur_in)
                assign(barrier_voltage, barrier_in)

                with for_(shot, 0, shot < num_shots, shot + 1):
                    dot_pair.initialize(
                        conditional_drive=True,
                    )

                    dot_pair.ramp_to_voltages(
                        voltages={
                            dot_pair.detuning_axis_name: detuning,
                            dot_pair.barrier_gate.id: barrier_voltage,
                        },
                        ramp_duration=ramp_dur,
                        duration=node.parameters.buffer_duration,
                    )

                    sensor = dot_pair.sensor_dots[0]
                    rr = sensor.readout_resonator
                    readout_length = rr.operations[node.parameters.operation].length
                    dot_pair.voltage_sequence.track_sticky_duration(readout_length)
                    align(rr.id, dot_pair.physical_channel.id)

                    rr.measure(
                        node.parameters.operation,
                        qua_vars=(I_var, Q_var),
                    )
                    save(I_var, I_st)
                    save(Q_var, Q_st)
                    align(rr.id, dot_pair.physical_channel.id)

                    dot_pair.voltage_sequence.apply_compensation_pulse()

        with stream_processing():
            I_st.buffer(pop_size * num_shots).save_all("I")
            Q_st.buffer(pop_size * num_shots).save_all("Q")

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    dot_pair_objects = [qp.quantum_dot_pair for qp in qubit_pairs]

    for gate_set_id in {
        dp.voltage_sequence.gate_set.id for dp in dot_pair_objects
    }:
        node.machine.reset_voltage_sequence(gate_set_id)

    for dot_pair in dot_pair_objects:
        if len(dot_pair.sensor_dots) != 1:
            raise ValueError(
                f"01_optimize_measurement_fidelity expects exactly one sensor "
                f"dot per pair; {dot_pair.id!r} has {len(dot_pair.sensor_dots)}"
            )

    node.namespace["qubit_pairs"] = qubit_pairs
    node.namespace["dot_pairs"] = dot_pair_objects

    node.namespace["qua_program"] = _build_qua_program(node, qubit_pairs[0])


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
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


def _compute_fidelities(
    I_data: np.ndarray,
    Q_data: np.ndarray,
) -> np.ndarray:
    """Compute readout fidelity for each candidate from I/Q shots.

    Parameters
    ----------
    I_data, Q_data : np.ndarray
        Shape ``(pop_size, num_shots)``.

    Returns
    -------
    np.ndarray
        Fidelity per candidate, shape ``(pop_size,)``.
    """
    import jax.numpy as jnp

    w, mu_iq = _pca_projector_np(I_data, Q_data)

    Z = (I_data - mu_iq[0]) * w[0] + (Q_data - mu_iq[1]) * w[1]
    Z_jnp = jnp.array(Z)

    mu1, sigma1, mu2, sigma2 = _vmap_em_two_gaussians(Z_jnp)

    fidelity, _ = _two_gaussian_fidelity_visibility(mu1, sigma1, mu2, sigma2)
    fidelity_np = np.asarray(fidelity)

    fidelity_np = np.where(np.isfinite(fidelity_np), fidelity_np, 0.5)

    return fidelity_np


# %% {Run_CMA-ES_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_cmaes_loop(node: QualibrationNode[Parameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    A dedicated QUA program is compiled and executed per pair so that only
    the pair under test is driven.  Within each pair's run the program
    stays alive via ``infinite_loop_``; each CMA-ES generation pushes new
    candidate parameters via ``push_to_input_stream`` — the OPX blocks at
    ``advance_input_stream`` until data arrives, so there is zero
    recompilation overhead between generations.  The job is cancelled once
    the optimisation finishes (or on error).
    """
    import time as _time

    qubit_pairs = node.namespace["qubit_pairs"]

    qmm = node.machine.connect()
    config = node.machine.generate_config()

    pop_size = node.parameters.population_size
    num_shots = node.parameters.num_shots
    param_names = ["detuning", "ramp_duration", "barrier_voltage"]

    x0 = np.array([
        node.parameters.detuning_initial,
        float(node.parameters.ramp_duration_initial),
        node.parameters.barrier_voltage_initial,
    ])
    bounds = [
        (node.parameters.detuning_min, node.parameters.detuning_max),
        (float(node.parameters.ramp_duration_min), float(node.parameters.ramp_duration_max)),
        (node.parameters.barrier_voltage_min, node.parameters.barrier_voltage_max),
    ]

    optimization_results = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES optimisation for {qp.name}...")

        qua_prog = _build_qua_program(node, qp)

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            result_handles = job.result_handles
            generation_counter = 0

            I_handle = result_handles.get("I")
            Q_handle = result_handles.get("Q")

            def evaluate_candidates(candidates: np.ndarray) -> np.ndarray:
                """Push one generation of candidates and fetch results."""
                nonlocal generation_counter

                for val in candidates[:, 0].tolist():
                    job.push_to_input_stream("detuning", val)
                ramp_dur_vals = [
                    max(16, int(round(v / 4.0)) * 4) for v in candidates[:, 1]
                ]
                for val in ramp_dur_vals:
                    job.push_to_input_stream("ramp_duration", val)
                for val in candidates[:, 2].tolist():
                    job.push_to_input_stream("barrier_voltage", val)

                target_count = generation_counter + 1
                while I_handle.count_so_far() < target_count:
                    _time.sleep(0.005)

                I_data = np.asarray(I_handle.fetch(generation_counter, flat_struct=True), dtype=np.float64).reshape(pop_size, num_shots)
                Q_data = np.asarray(Q_handle.fetch(generation_counter, flat_struct=True), dtype=np.float64).reshape(pop_size, num_shots)

                generation_counter += 1
                return _compute_fidelities(I_data, Q_data)

            try:
                opt_result = run_cmaes_optimization(
                    evaluate_fn=evaluate_candidates,
                    param_names=param_names,
                    x0=x0,
                    sigma0=node.parameters.sigma0,
                    bounds=bounds,
                    population_size=pop_size,
                    max_generations=node.parameters.max_generations,
                    tolx=node.parameters.tolx,
                    tolfun=node.parameters.tolfun,
                    log_callable=node.log,
                    progress_prefix=qp.name,
                    log_each_generation=node.parameters.cmaes_log_each_generation,
                )
                optimization_results[qp.name] = opt_result
            finally:
                job.cancel()

    node.namespace["optimization_results"] = optimization_results
    node.results["optimization_results"] = {
        name: result.to_dict() for name, result in optimization_results.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = _resolve_qubit_pairs(node)
    node.namespace["optimization_results"] = {
        name: OptimizationResult.from_dict(d)
        for name, d in node.results["optimization_results"].items()
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
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
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Generate convergence and parameter-evolution plots (per qubit pair)."""
    opt_results = node.namespace["optimization_results"]

    fig_conv = plot_convergence(opt_results)
    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    node.results["figures"] = {
        "convergence": fig_conv,
        "parameter_evolution": fig_params,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write each qubit pair's optimal detuning, barrier voltage, and ramp duration to QUAM."""
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
            optimal_detuning = best["detuning"]
            optimal_barrier_voltage = best["barrier_voltage"]
            optimal_ramp_duration = max(
                16, int(round(best["ramp_duration"] / 4.0)) * 4
            )

            dot_pair = qp.quantum_dot_pair

            dot_pair.add_point(
                VoltagePointName.MEASURE,
                voltages={
                    dot_pair.detuning_axis_name: optimal_detuning,
                    dot_pair.barrier_gate.id: optimal_barrier_voltage,
                },
                duration=node.parameters.buffer_duration,
                replace_existing_point=True,
            )

            init_macro = dot_pair.macros.get("initialize")
            if init_macro is not None and hasattr(init_macro, "update"):
                init_macro.update(ramp_duration=optimal_ramp_duration)
            else:
                node.log(
                    f"  {qp.name}: no updatable initialise macro found on "
                    f"{dot_pair.name}"
                )

            node.log(
                f"  {qp.name}: detuning = {optimal_detuning:.6g} V, "
                f"barrier_voltage = {optimal_barrier_voltage:.6g} V, "
                f"ramp_duration = {optimal_ramp_duration} ns"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
