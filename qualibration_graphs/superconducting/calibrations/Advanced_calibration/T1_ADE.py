# %% {Imports}
from dataclasses import asdict

from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.T1_ADE import (
    DENOM_SQ_FLOOR,
    GAMMA_ADAPTIVE_FLOOR,
    LN_ARG_FLOOR,
    Parameters,
    SAFE_CEILING,
    SQRT_ARG_FLOOR,
    TIME_SCALE_US,
    fetch_raw_dataset,
    fit_raw_data,
    plot_raw_data_with_fit,
    process_raw_dataset,
)

# %% {Node initialisation}
description = """
        T1 ADE TRACKING
Tracks T1 vs laboratory time using Analytical Decay Estimation (ADE) with on-FPGA uncertainty
propagation (arXiv:2602.11912 Appendix F). Each repetition measures P(|1>) at t in
{t0, t0+dt, t0+3dt} with interleaved shots so P0/P1/P3 sample the same lab-time window --
important when T1 drifts during a repetition. SPAM offset and amplitude cancel in the ratio;
no confusion matrix is required.

Point estimate (per repetition):
    c     = (P3 - P0) / (P1 - P0)
    x     = sqrt(c - 3/4) - 1/2
    gamma = -ln(x) / dt          [1/us on host; streamed as estimated_gamma]

Uncertainty (arXiv:2602.11912 Sec II):

On FPGA (primary sigma, streamed as sigma_gamma):
    sigma_Pi    = sqrt(Pi * (1 - Pi) / n_avg)   binomial shot noise, independent per delay
    dgamma/dPi  = (dgamma/dc) * (dc/dPi)        chain rule through c; dc/dP0 uses the full
                                                quotient rule (P0 in num. and denom.)
    sigma_gamma = sqrt(sum_i (dgamma/dPi * sigma_Pi)^2)

Rate space is kept on the FPGA (gamma, sigma_gamma) to avoid 1/gamma and the gamma -> T1
Jacobian in QUA fixed-point. Host converts:
    T1       = 1 / gamma
    sigma_T1 = sigma_gamma / gamma^2

Bootstrapped on host (comparison only; n_bootstrap parameter, not on hardware):
    For each repetition, draw n_bootstrap resamples of the n_avg per-shot outcomes at each
    delay (with replacement; independent indices for P0, P1, P3). Re-run the ADE formula on
    each draw to get gamma_boot, then T1_boot = 1 / gamma_boot. Report
    sigma_T1_boot = (P84(T1_boot) - P16(T1_boot)) / 2.
    Requires per-shot streams (shots0_/1_/3_) -- same T1 point estimate as FPGA; only sigma differs.

Numerical guards: sqrt/ln arguments and denom^2 are floored from the QUA fixed range [-8, 8).
Derivatives ignore those clip floors (valid in the interior). Host flags clipped repetitions and
masks them in plots.

Optional adaptivity: if adaptive_dt is True, dt for the next repetition is set from the running
gamma estimate, clipped to [min_dt_ns, max_dt_ns].

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a and 04b).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

Outputs:
    - T1 vs laboratory time with FPGA analytical and bootstrap sigma bands
    - ADE exponential fit at the lowest-sigma repetition per qubit
    - Time-to-decision statistics and full dataset in node.results["ds_fit"]
"""

node = QualibrationNode[Parameters, Quam](
    name="T1_ADE",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubits = ["qA1"]
    node.parameters.adaptive_dt = False
    node.parameters.min_dt_ns = 16
    node.parameters.max_dt_ns = 150_000
    node.parameters.alpha = 1.0
    node.parameters.t1_guess_us = 40.0
    node.parameters.n_avg_per_point = 50
    node.parameters.n_bootstrap = 300
    node.parameters.reset_max_attempts = 1
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the ADE QUA program and store it in the node namespace."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_reps = node.parameters.num_repetitions
    t0_cycles = node.parameters.t0_ns // 4
    initial_dt_cycles = max(1, round(node.parameters.alpha * node.parameters.t1_guess_us * 1000 / 4))
    min_dt_cycles = max(1, node.parameters.min_dt_ns // 4)
    max_dt_cycles = max(min_dt_cycles + 1, node.parameters.max_dt_ns // 4)
    inv_n_avg = 1.0 / node.parameters.n_avg_per_point

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables()

        state = [declare(int) for _ in range(num_qubits)]
        estimated_gamma_st = [declare_output_stream() for _ in range(num_qubits)]
        sigma_gamma_st = [declare_output_stream() for _ in range(num_qubits)]
        dt_used_st = [declare_output_stream() for _ in range(num_qubits)]
        P0_st = [declare_output_stream() for _ in range(num_qubits)]
        P1_st = [declare_output_stream() for _ in range(num_qubits)]
        P3_st = [declare_output_stream() for _ in range(num_qubits)]
        shots0_st = [declare_output_stream() for _ in range(num_qubits)]
        shots1_st = [declare_output_stream() for _ in range(num_qubits)]
        shots3_st = [declare_output_stream() for _ in range(num_qubits)]

        for i, qubit in enumerate(qubits):
            acc0 = declare(int)
            acc1 = declare(int)
            acc3 = declare(int)
            shot = declare(int)

            P0 = declare(fixed)
            P1 = declare(fixed)
            P3 = declare(fixed)
            c = declare(fixed)
            sqrt_arg = declare(fixed)
            xval = declare(fixed)
            dt_scaled = declare(fixed)
            gamma1_est = declare(fixed)
            denom = declare(fixed)
            denom_sq = declare(fixed)
            x_plus_half = declare(fixed)
            dgamma_dc = declare(fixed)
            dgamma_dP0 = declare(fixed)
            dgamma_dP1 = declare(fixed)
            dgamma_dP3 = declare(fixed)
            sigma_P0 = declare(fixed)
            sigma_P1 = declare(fixed)
            sigma_P3 = declare(fixed)
            term0 = declare(fixed)
            term1 = declare(fixed)
            term2 = declare(fixed)
            sigma_gamma = declare(fixed)
            gamma_safe = declare(fixed)
            T1_est_scaled = declare(fixed)

            delta_t_cycles = declare(int, value=initial_dt_cycles)

            node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_reps, n + 1):
                save(n, n_st)
                assign(acc0, 0)
                assign(acc1, 0)
                assign(acc3, 0)

                with for_(shot, 0, shot < node.parameters.n_avg_per_point, shot + 1):
                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    )
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.wait(t0_cycles)
                    qubit.align()
                    qubit.readout_state(state[i])
                    assign(acc0, acc0 + state[i])
                    save(state[i], shots0_st[i])

                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    )
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.wait(t0_cycles + delta_t_cycles)
                    qubit.align()
                    qubit.readout_state(state[i])
                    assign(acc1, acc1 + state[i])
                    save(state[i], shots1_st[i])

                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    )
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.wait(t0_cycles + 3 * delta_t_cycles)
                    qubit.align()
                    qubit.readout_state(state[i])
                    assign(acc3, acc3 + state[i])
                    save(state[i], shots3_st[i])

                assign(P0, Cast.mul_fixed_by_int(inv_n_avg, acc0))
                assign(P1, Cast.mul_fixed_by_int(inv_n_avg, acc1))
                assign(P3, Cast.mul_fixed_by_int(inv_n_avg, acc3))
                save(P0, P0_st[i])
                save(P1, P1_st[i])
                save(P3, P3_st[i])

                assign(c, Math.div(P3 - P0, P1 - P0))
                assign(sqrt_arg, SQRT_ARG_FLOOR + Math.relu((c - 0.75) - SQRT_ARG_FLOOR))
                assign(xval, Math.sqrt(sqrt_arg) - 0.5)
                assign(xval, LN_ARG_FLOOR + Math.relu(xval - LN_ARG_FLOOR))
                assign(dt_scaled, Cast.mul_fixed_by_int(4e-3 / TIME_SCALE_US, delta_t_cycles))
                assign(gamma1_est, (-Math.div(Math.ln(xval), dt_scaled)) * (1.0 / TIME_SCALE_US))
                save(gamma1_est, estimated_gamma_st[i])
                save(delta_t_cycles, dt_used_st[i])

                assign(denom, P1 - P0)
                assign(denom_sq, denom * denom)
                assign(denom_sq, DENOM_SQ_FLOOR + Math.relu(denom_sq - DENOM_SQ_FLOOR))
                assign(x_plus_half, xval + 0.5)

                assign(dgamma_dc, Math.div(-1.0, 2.0 * xval * x_plus_half * dt_scaled))
                assign(dgamma_dc, dgamma_dc * (1.0 / TIME_SCALE_US))
                assign(dgamma_dc, -SAFE_CEILING + Math.relu(dgamma_dc + SAFE_CEILING))
                assign(dgamma_dc, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dc))

                assign(dgamma_dP0, dgamma_dc * Math.div(P3 - P1, denom_sq))
                assign(dgamma_dP1, dgamma_dc * Math.div(-(P3 - P0), denom_sq))
                assign(dgamma_dP3, dgamma_dc * Math.div(denom, denom_sq))

                assign(dgamma_dP0, -SAFE_CEILING + Math.relu(dgamma_dP0 + SAFE_CEILING))
                assign(dgamma_dP0, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP0))
                assign(dgamma_dP1, -SAFE_CEILING + Math.relu(dgamma_dP1 + SAFE_CEILING))
                assign(dgamma_dP1, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP1))
                assign(dgamma_dP3, -SAFE_CEILING + Math.relu(dgamma_dP3 + SAFE_CEILING))
                assign(dgamma_dP3, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP3))

                assign(sigma_P0, Math.sqrt(P0 * (1.0 - P0) * inv_n_avg))
                assign(sigma_P1, Math.sqrt(P1 * (1.0 - P1) * inv_n_avg))
                assign(sigma_P3, Math.sqrt(P3 * (1.0 - P3) * inv_n_avg))

                assign(term0, dgamma_dP0 * sigma_P0)
                assign(term0, term0 * term0)
                assign(term1, dgamma_dP1 * sigma_P1)
                assign(term1, term1 * term1)
                assign(term2, dgamma_dP3 * sigma_P3)
                assign(term2, term2 * term2)
                assign(sigma_gamma, Math.sqrt(term0 + term1 + term2))
                save(sigma_gamma, sigma_gamma_st[i])

                qubit.xy.play("x90", amplitude_scale=0, duration=4, timestamp_stream=f"time_stamp{i+1}")

                if node.parameters.adaptive_dt:
                    assign(gamma_safe, GAMMA_ADAPTIVE_FLOOR + Math.relu(gamma1_est - GAMMA_ADAPTIVE_FLOOR))
                    assign(T1_est_scaled, Math.div(1.0, gamma_safe * TIME_SCALE_US))
                    _delta_t_int_const = round(node.parameters.alpha * 250 * TIME_SCALE_US)
                    assign(delta_t_cycles, Cast.mul_int_by_fixed(_delta_t_int_const, T1_est_scaled))
                    with if_(delta_t_cycles < min_dt_cycles):
                        assign(delta_t_cycles, min_dt_cycles)
                    with if_(delta_t_cycles > max_dt_cycles):
                        assign(delta_t_cycles, max_dt_cycles)

        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                estimated_gamma_st[i].buffer(n_reps).save(f"estimated_gamma{i + 1}")
                sigma_gamma_st[i].buffer(n_reps).save(f"sigma_gamma{i + 1}")
                dt_used_st[i].buffer(n_reps).save(f"dt_used{i + 1}")
                P0_st[i].buffer(n_reps).save(f"P0{i + 1}")
                P1_st[i].buffer(n_reps).save(f"P1{i + 1}")
                P3_st[i].buffer(n_reps).save(f"P3{i + 1}")
                shots0_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"shots0_{i + 1}")
                shots1_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"shots1_{i + 1}")
                shots3_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"shots3_{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters,
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program, show live progress on n, then fetch all streams."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    n_reps = node.parameters.num_repetitions
    n_avg = node.parameters.n_avg_per_point
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            progress_counter(
                results.fetch_all()[0],
                n_reps,
                start_time=results.start_time,
            )
        node.log(job.execution_report())
        node.results["ds_raw"] = fetch_raw_dataset(job, qubits, n_reps, n_avg)

# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw streams and extract per-qubit ADE fit results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    ds = node.results["ds_fit"]
    qubits = node.namespace["qubits"]
    node.results["time_to_decision_ms"] = ds.attrs["time_to_decision_ms"]
    node.results["sigma_gamma_fpga"] = {
        q.name: ds.sigma_gamma.sel(qubit=q.name).values for q in qubits
    }
    node.results["T1_analytical_sigma_us"] = {k: v.sigma_T1_us for k, v in fit_results.items()}
    node.results["T1_clipped"] = {k: v.clipped for k, v in fit_results.items()}
    node.results["T1_bootstrap_sigma_us"] = {k: v.sigma_T1_boot_us for k, v in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot T1 vs lab time and the ADE wait-time fit at the best-sigma repetition."""
    figs = plot_raw_data_with_fit(node)
    node.results["figures"] = {
        "T1_vs_lab_time": figs["T1_vs_lab_time"],
        "T1_vs_lab_time_bootstrap": figs["T1_vs_lab_time_bootstrap"],
        "wait_time_image": figs["wait_time_image"],
    }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
