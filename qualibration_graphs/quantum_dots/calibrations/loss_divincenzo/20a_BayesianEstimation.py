# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.bayesian_estimation import Parameters
from calibration_utils.bayesian_estimation.analysis import map_estimates_from_raw
from calibration_utils.bayesian_estimation.idle_grid import sweep_from_parameters
from calibration_utils.bayesian_estimation.plotting import (
    plot_pf_posterior,
    plot_pf_posterior_single_rep_all_qubits_with_map_track,
)
from calibration_utils.bayesian_estimation.simulation import simulate_bayesian_ramsey_dataset
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.annotation import annotate_node_figures, stamp_snapshot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot


def _readout_quantum_dot_pair_for_qubit(qubit, machine):
    preferred = getattr(qubit, "preferred_readout_quantum_dot", None)
    if preferred is None:
        raise ValueError(
            f"Qubit {qubit.id!r} has no preferred_readout_quantum_dot; "
            "set it to the partner dot used for PSB readout (same as single-qubit measure)."
        )
    pair_name = machine.find_quantum_dot_pair(qubit.quantum_dot.id, preferred)
    if pair_name is None:
        raise ValueError(f"No QuantumDotPair registered for dots {qubit.quantum_dot.id!r} and {preferred!r}.")
    return machine.quantum_dot_pairs[pair_name]


# %% {Node initialisation}
description = """
        BAYESIAN FREQUENCY ESTIMATION (Ramsey-style, adaptive posterior on a grid)
Sweeps idle time between two π/2 pulses while accumulating a discrete posterior P(f)
over a hypothesis grid in MHz around ``detuning`` ± ``f_span_in_MHz`` (step ``f_step_in_MHz``).
When ``derive_idle_times`` is True (default), the delay step Δτ (ns) is chosen so the delay-axis
Nyquist frequency f_Nyquist = 500/Δτ MHz satisfies f_Nyquist ≥ ``nyquist_margin`` × max|f| on
that grid (default ``nyquist_margin`` is 20). The longest delay follows ``f_step_in_MHz`` (see
``idle_grid.max_idle_ns_from_frequency_step``) and is capped by ``max_wait_time_in_ns`` (defaults
are aligned so the cap does not truncate that bound for the stock ``f_step_in_MHz``).

Each binary readout updates P(f) ∝ (0.5 + r_k (α + β cos2π f t)) P(f) with r_k = ±0.5 from
discriminated spin outcomes; cosine-only Ramsey cannot separate +f and −f on the grid. After each
shot the prior for f is reset to uniform before the next shot.

Readout uses ``QuantumDotPair.readout_state`` (sensor IQ vs calibrated threshold), so each
qubit must have ``preferred_readout_quantum_dot`` set and a matching registered dot pair.

Prerequisites:
    - Calibrated initialization / measurement voltage points and X90 pulses.
    - IQ readout calibration (threshold / projector) for the relevant sensor dot pair.

Analysis:
    - MAP estimate of f (MHz on the hypothesis grid) from the posterior at the longest idle time (mean over repetitions).

State update (optional):
    - If ``apply_map_estimate_to_larmor`` is True, adds the MAP f in MHz to ``qubit.larmor_frequency``.

Python synthetic data:
    - If ``synthetic_example_f_mhz`` is set, the node skips OPX execute/simulate, fills ``ds_raw`` from
    ``calibration_utils.bayesian_estimation.simulation``, and adds ``figure_single_rep`` (one repetition,
    argmax f(τ) on the posterior image).
"""


node = QualibrationNode[Parameters, Quam](
    name="20a_BayesianEstimation",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.synthetic_example_f_mhz = 0.05
    node.parameters.qubits = ["q1"]
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build sweep axes and QUA program for Bayesian Ramsey posterior streaming."""
    machine = node.machine
    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots
    p = node.parameters

    v_f, tau_ns, idle_times, sweep_meta = sweep_from_parameters(p)
    if len(idle_times) == 0:
        raise ValueError("Empty idle time sweep; check detuning, f_span_in_MHz, f_step_in_MHz, and idle-time bounds.")
    if len(v_f) == 0:
        raise ValueError("Empty frequency grid; check detuning, f_span_in_MHz, and f_step_in_MHz.")

    readout_pairs = [_readout_quantum_dot_pair_for_qubit(q, machine) for q in qubits]

    node.namespace["v_f"] = v_f
    node.namespace["tau_ns"] = tau_ns
    node.namespace["idle_sweep_meta"] = sweep_meta
    node.namespace["readout_pairs"] = readout_pairs
    node.log(f"Bayesian idle sweep: {sweep_meta}")

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "repetition": xr.DataArray(np.arange(n_avg)),
        "tau": xr.DataArray(tau_ns, attrs={"long_name": "idle time", "units": "ns"}),
        "frequency": xr.DataArray(v_f, attrs={"long_name": "hypothesis frequency", "units": "MHz"}),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        n = declare(int)
        t = declare(int)
        f_idx = declare(int)

        frequencies = declare(fixed, value=v_f.tolist())
        Pf = declare(fixed, value=(np.ones(len(v_f)) / len(v_f)).tolist())
        alpha = declare(fixed, value=p.alpha)
        beta = declare(fixed, value=p.beta)

        norm = declare(fixed)
        rk = declare(fixed)
        t_sample = declare(fixed)
        C = declare(fixed)
        p_uniform = declare(fixed, value=float(1.0 / len(v_f)))

        n_st = declare_output_stream()

        state = [declare(int) for _ in qubits]
        state_st = [declare_stream() for _ in qubits]
        Pf_st = [declare_stream() for _ in qubits]

        for i, qubit in enumerate(qubits):
            qubit.xy.update_frequency(
                qubit.xy.intermediate_frequency + p.detuning
            )  # set detuning from qubit Larmor frequency
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(t, idle_times)):  # idle times here in clock cycles
                    qubit.initialize()
                    qubit.x90()  # TODO: does this take care of track_sticky_duration?
                    qubit.wait(t)
                    qubit.voltage_sequence.track_sticky_duration(t)
                    qubit.x90()  # TODO: this ends up ideally with |1> state if not detuned

                    a2 = qubit.measure()
                    qubit.voltage_sequence.ramp_to_zero()
                    # Bayesian estimation starts here
                    assign(state[i], Cast.to_int(a2))
                    save(state[i], state_st[i])

                    assign(rk, Cast.to_fixed(state[i]) - 0.5)
                    assign(t_sample, Cast.mul_fixed_by_int(1e-3, t * 4))  # t_sample in microseconds

                    with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                        assign(
                            C, Math.cos2pi(frequencies[f_idx] * t_sample)
                        )  # frequencies here in MHz and t_sample in microseconds
                        assign(
                            Pf[f_idx],
                            (0.5 + rk * (alpha + beta * C)) * Pf[f_idx],
                        )

                    assign(norm, Cast.to_fixed(1.0 / Math.sum(Pf)))
                    with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                        assign(Pf[f_idx], norm * Pf[f_idx])
                        save(Pf[f_idx], Pf_st[i])

                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(Pf[f_idx], p_uniform)

        with stream_processing():
            n_st.save("n")
            for i in range(len(qubits)):
                Pf_st[i].buffer(n_avg, len(idle_times), len(v_f)).save(f"Pf{i + 1}")
                state_st[i].buffer(n_avg, len(idle_times)).save(f"state{i + 1}")


# %% {Synthetic_data}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.synthetic_example_f_mhz is None)
def inject_synthetic_bayesian_data(node: QualibrationNode[Parameters, Quam]):
    """Fill ``ds_raw`` from the analytic simulator (no OPX acquisition)."""
    f_true = float(node.parameters.synthetic_example_f_mhz)
    qnames = [q.name for q in node.namespace["qubits"]]
    node.results["ds_raw"] = simulate_bayesian_ramsey_dataset(
        node.parameters,
        f_true,
        qnames,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.synthetic_example_f_mhz is not None
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.synthetic_example_f_mhz is not None
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch raw data into ``ds_raw``."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):

    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """MAP frequency from posterior at longest tau (mean over repetitions)."""
    v_f = node.namespace.get("v_f")
    if v_f is None:
        v_f, _, _, _ = sweep_from_parameters(node.parameters)
    qnames = [q.name for q in node.namespace["qubits"]]
    fit_results, estimates = map_estimates_from_raw(
        node.results["ds_raw"],
        qnames,
        v_f,
    )
    node.results["fit_results"] = fit_results
    node.results["estimates"] = estimates
    node.outcomes = {q: ("successful" if fit_results.get(q, {}).get("success") else "failed") for q in qnames}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Posterior P(f) vs idle time (mean over repetitions)."""
    tau_ns = node.namespace.get("tau_ns")
    v_f = node.namespace.get("v_f")
    if tau_ns is None or v_f is None:
        v_f, tau_ns, _, _ = sweep_from_parameters(node.parameters)

    qnames = [q.name for q in node.namespace["qubits"]]
    fig = plot_pf_posterior(
        node.results["ds_raw"],
        qnames,
        v_f,
        tau_ns,
        node.results.get("fit_results"),
    )
    node.results["figure"] = fig

    if node.parameters.synthetic_example_f_mhz is not None:
        node.results["figure_single_rep"] = plot_pf_posterior_single_rep_all_qubits_with_map_track(
            node.results["ds_raw"],
            qnames,
            v_f,
            tau_ns,
            repetition_index=0,
        )

    annotate_node_figures(node)

    fig_single = node.results.get("figure_single_rep")
    if fig_single is not None:
        stamp_snapshot(fig_single, node.snapshot_idx, node.name)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
