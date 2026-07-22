# %% {Imports}
from dataclasses import asdict
import math

import numpy as np
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_idle_times_in_clock_cycles, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.T1_ADE import (
    Parameters,
    fetch_raw_dataset,
    fit_raw_data,
    plot_raw_data_with_fit,
    process_raw_dataset,
)

# QUA `fixed` ~ [-8, 8). Clip floors for ADE gamma / sigma on FPGA.
QUA_FIXED_MAX = 8.0
SAFE_CEILING = QUA_FIXED_MAX - 2.0
TIME_SCALE_US = 4.0 * QUA_FIXED_MAX
LN_ARG_FLOOR = math.exp(-QUA_FIXED_MAX)
SQRT_ARG_FLOOR = (LN_ARG_FLOOR + 0.5) ** 2
GAMMA_ADAPTIVE_FLOOR = 1.0 / TIME_SCALE_US
DENOM_SQ_FLOOR = 1.0 / SAFE_CEILING
GAMMA_FLOOR = 1e-4
DENOM_MIN = math.sqrt(DENOM_SQ_FLOOR)

ADE_CLIP_FLOORS = {
    "ln_arg_floor": LN_ARG_FLOOR,
    "sqrt_arg_floor": SQRT_ARG_FLOOR,
    "denom_min": DENOM_MIN,
    "gamma_floor": GAMMA_FLOOR,
}


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
    gamma1 = -ln(x) / dt          [1/us on host; streamed as gamma1]

Uncertainty (arXiv:2602.11912 Sec II):

On FPGA (primary sigma, streamed as sigma_gamma1):
    sigma_Pi    = sqrt(Pi * (1 - Pi) / n_avg)   binomial shot noise, independent per delay
    dgamma/dPi  = (dgamma/dc) * (dc/dPi)        chain rule through c; dc/dP0 uses the full
                                                quotient rule (P0 in num. and denom.)
    sigma_gamma1 = sqrt(sum_i (dgamma/dPi * sigma_Pi)^2)

Rate space is kept on the FPGA (gamma1, sigma_gamma1) to avoid 1/gamma1 and the gamma1 -> T1
Jacobian in QUA fixed-point. Host converts:
    T1       = 1 / gamma1
    sigma_T1 = sigma_gamma1 / gamma1^2

Bootstrapped on host (comparison only; n_bootstrap parameter, not on hardware):
    For each repetition, draw n_bootstrap resamples of the n_avg per-shot outcomes at each
    delay (with replacement; independent indices for P0, P1, P3). Re-run the ADE formula on
    each draw to get gamma_boot, then T1_boot = 1 / gamma_boot. Report
    sigma_T1_boot = (P84(T1_boot) - P16(T1_boot)) / 2.
    Requires per-shot streams (P0_shots / P1_shots / P3_shots) -- same T1 point estimate as FPGA; only sigma differs.

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
    - Optional mid-run conventional T1 sweep (``measure_conventional_t1``)
    - Time-to-decision statistics and full dataset in node.results["ds_fit"]
"""

node = QualibrationNode[Parameters, Quam](
    name="T1_ADE",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)
node.namespace["ade_clip_floors"] = ADE_CLIP_FLOORS


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the ADE QUA program and store it in the node namespace."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    measure_conventional_t1 = node.parameters.measure_conventional_t1

    t1_idle_times: list[int] = []
    if measure_conventional_t1:
        t1_idle_cycles = get_idle_times_in_clock_cycles(node.parameters)
        t1_idle_times = t1_idle_cycles.tolist()
        node.namespace["t1_conventional_delays_ns"] = 4 * t1_idle_cycles

    n_reps = node.parameters.num_repetitions
    t0_cycles = node.parameters.t0_ns // 4  # shortest ADE delay (P0)
    initial_dt_cycles = max(1, round(node.parameters.alpha * node.parameters.t1_guess_us * 1000 / 4)) # initial dt from T1 guess
    min_dt_cycles = max(1, node.parameters.min_dt_ns // 4)  # adaptive-dt clip bounds
    max_dt_cycles = max(min_dt_cycles + 1, node.parameters.max_dt_ns // 4)
    inv_n_avg = 1.0 / node.parameters.n_avg_per_point  # FPGA average over interleaved shots

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables()

        state = [declare(int) for _ in range(num_qubits)] # state of the qubit
        gamma1_st = [declare_output_stream() for _ in range(num_qubits)] # gamma1 stream
        sigma_gamma1_st = [declare_output_stream() for _ in range(num_qubits)] # sigma_gamma1 stream
        dt_used_st = [declare_output_stream() for _ in range(num_qubits)] # dt used stream
        P0_st = [declare_output_stream() for _ in range(num_qubits)] # P(|1>) at t0 stream
        P1_st = [declare_output_stream() for _ in range(num_qubits)] # P(|1>) at t0 + dt stream
        P3_st = [declare_output_stream() for _ in range(num_qubits)] # P(|1>) at t0 + 3*dt stream
        P0_shots_st = [declare_output_stream() for _ in range(num_qubits)] # per-shot P(|1>) at t0 stream
        P1_shots_st = [declare_output_stream() for _ in range(num_qubits)] # per-shot P(|1>) at t0 + dt stream
        P3_shots_st = [declare_output_stream() for _ in range(num_qubits)] # per-shot P(|1>) at t0 + 3*dt stream
        if measure_conventional_t1:
            t1_state_st = [declare_output_stream() for _ in range(num_qubits)] # state of the qubit at the end of the conventional T1 measurement

        for i, qubit in enumerate(qubits):
            acc0 = declare(int) # accumulator for P(|1>) at t0
            acc1 = declare(int) # accumulator for P(|1>) at t0 + dt
            acc3 = declare(int) # accumulator for P(|1>) at t0 + 3*dt
            shot = declare(int) # shot index
            if measure_conventional_t1:
                t1_shot = declare(int) # shot index for the conventional T1 measurement
                t = declare(int) # idle time

            P0 = declare(fixed) # P(|1>) at t0
            P1 = declare(fixed) # P(|1>) at t0 + dt
            P3 = declare(fixed) # P(|1>) at t0 + 3*dt
            c = declare(fixed) # c = (P3 - P0) / (P1 - P0)
            sqrt_arg = declare(fixed) # sqrt(c - 3/4)
            xval = declare(fixed) # x = sqrt(c - 3/4) - 1/2
            dt_scaled = declare(fixed) # dt in scaled units
            gamma1_est = declare(fixed)
            denom = declare(fixed) # denominator of the ADE formula
            denom_sq = declare(fixed) # denominator squared
            x_plus_half = declare(fixed) # x + 0.5
            dgamma_dc = declare(fixed) # dgamma/dc
            dgamma_dP0 = declare(fixed) # dgamma/dP0
            dgamma_dP1 = declare(fixed) # dgamma/dP1
            dgamma_dP3 = declare(fixed) # dgamma/dP3
            sigma_P0 = declare(fixed) # sigma of P(|1>) at t0
            sigma_P1 = declare(fixed) # sigma of P(|1>) at t0 + dt
            sigma_P3 = declare(fixed) # sigma of P(|1>) at t0 + 3*dt
            term0 = declare(fixed) # term0 = dgamma_dP0 * sigma_P0
            term1 = declare(fixed) # term1 = dgamma_dP1 * sigma_P1
            term2 = declare(fixed) # term2 = dgamma_dP3 * sigma_P3
            sigma_gamma1 = declare(fixed) # sigma of gamma1
            gamma_safe = declare(fixed) # gamma floored
            T1_est_scaled = declare(fixed) # T1 estimated in scaled units

            delta_t_cycles = declare(int, value=initial_dt_cycles) # initial dt in cycles

            node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_reps, n + 1):
                save(n, n_st)
                assign(acc0, 0)
                assign(acc1, 0)
                assign(acc3, 0)

                with for_(shot, 0, shot < node.parameters.n_avg_per_point, shot + 1): # average over n_avg_per_point shots
                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    ) # reset the qubit and align it
                    qubit.align()
                    qubit.xy.play("x180") # play the x180 pulse
                    qubit.wait(t0_cycles) # wait for t0_cycles
                    qubit.align()
                    qubit.readout_state(state[i]) # readout the state of the qubit
                    assign(acc0, acc0 + state[i]) # accumulate the state of the qubit
                    save(state[i], P0_shots_st[i]) # save the state of the qubit to the stream

                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    ) # reset the qubit and align it
                    qubit.align()
                    qubit.xy.play("x180") # play the x180 pulse
                    qubit.wait(t0_cycles + delta_t_cycles) # wait for t0_cycles + delta_t_cycles
                    qubit.align() # align the qubit
                    qubit.readout_state(state[i]) # readout the state of the qubit
                    assign(acc1, acc1 + state[i]) # accumulate the state of the qubit
                    save(state[i], P1_shots_st[i]) # save the state of the qubit to the stream

                    qubit.reset_qubit_active(
                        pi_pulse_name="x180",
                        readout_pulse_name="readout",
                        max_attempts=node.parameters.reset_max_attempts,
                    ) # reset the qubit and align it
                    qubit.align()
                    qubit.xy.play("x180") # play the x180 pulse
                    qubit.wait(t0_cycles + 3 * delta_t_cycles) # wait for t0_cycles + 3 * delta_t_cycles
                    qubit.align() # align the qubit
                    qubit.readout_state(state[i]) # readout the state of the qubit
                    assign(acc3, acc3 + state[i]) # accumulate the state of the qubit
                    save(state[i], P3_shots_st[i]) # save the state of the qubit to the stream

                assign(P0, Cast.mul_fixed_by_int(inv_n_avg, acc0)) # average the state of the qubit at t0
                assign(P1, Cast.mul_fixed_by_int(inv_n_avg, acc1)) # average the state of the qubit at t0 + dt
                assign(P3, Cast.mul_fixed_by_int(inv_n_avg, acc3)) # average the state of the qubit at t0 + 3*dt
                save(P0, P0_st[i]) # save the state of the qubit at t0 to the stream
                save(P1, P1_st[i]) # save the state of the qubit at t0 + dt to the stream
                save(P3, P3_st[i]) # save the state of the qubit at t0 + 3*dt to the stream

                # callaue gamma from the ADE formula
                assign(c, Math.div(P3 - P0, P1 - P0)) # calculate the c value
                assign(sqrt_arg, SQRT_ARG_FLOOR + Math.relu((c - 0.75) - SQRT_ARG_FLOOR)) # calculate the sqrt_arg value
                assign(xval, Math.sqrt(sqrt_arg) - 0.5) # calculate the xval value
                assign(xval, LN_ARG_FLOOR + Math.relu(xval - LN_ARG_FLOOR)) # make sure xval respects the QUA fixed range [-8, 8)
                assign(dt_scaled, Cast.mul_fixed_by_int(4e-3 / TIME_SCALE_US, delta_t_cycles)) # calculate the dt_scaled value
                assign(gamma1_est, (-Math.div(Math.ln(xval), dt_scaled)) * (1.0 / TIME_SCALE_US)) # calculate the gamma1_est value
                save(gamma1_est, gamma1_st[i])
                save(delta_t_cycles, dt_used_st[i]) # save the delta_t_cycles value to the stream

                assign(denom, P1 - P0) # calculate the denom value
                assign(denom_sq, denom * denom) # calculate the denom_sq value
                assign(denom_sq, DENOM_SQ_FLOOR + Math.relu(denom_sq - DENOM_SQ_FLOOR)) # Make sure denom_sq respects the QUA fixed range [-8, 8)
                assign(x_plus_half, xval + 0.5) # calculate the x_plus_half value

                assign(dgamma_dc, Math.div(-1.0, 2.0 * xval * x_plus_half * dt_scaled)) # calculate the dgamma_dc value
                assign(dgamma_dc, dgamma_dc * (1.0 / TIME_SCALE_US)) # calculate the dgamma_dc value
                # Make sure dgamma_dc respects the QUA fixed range [-8, 8)
                assign(dgamma_dc, -SAFE_CEILING + Math.relu(dgamma_dc + SAFE_CEILING)) 
                assign(dgamma_dc, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dc)) 

                assign(dgamma_dP0, dgamma_dc * Math.div(P3 - P1, denom_sq)) # calculate the dgamma_dP0 value
                assign(dgamma_dP1, dgamma_dc * Math.div(-(P3 - P0), denom_sq)) # calculate the dgamma_dP1 value
                assign(dgamma_dP3, dgamma_dc * Math.div(denom, denom_sq)) # calculate the dgamma_dP3 value

                # Make sure dgamma_dP0, dgamma_dP1, and dgamma_dP3 respect the QUA fixed range [-8, 8)
                assign(dgamma_dP0, -SAFE_CEILING + Math.relu(dgamma_dP0 + SAFE_CEILING)) 
                assign(dgamma_dP0, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP0))
                assign(dgamma_dP1, -SAFE_CEILING + Math.relu(dgamma_dP1 + SAFE_CEILING))
                assign(dgamma_dP1, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP1))
                assign(dgamma_dP3, -SAFE_CEILING + Math.relu(dgamma_dP3 + SAFE_CEILING))
                assign(dgamma_dP3, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dP3))

                # Calculate the sigma of the P(|1>) at t0, t0 + dt, and t0 + 3*dt
                assign(sigma_P0, Math.sqrt(P0 * (1.0 - P0) * inv_n_avg))
                assign(sigma_P1, Math.sqrt(P1 * (1.0 - P1) * inv_n_avg))
                assign(sigma_P3, Math.sqrt(P3 * (1.0 - P3) * inv_n_avg))

                # Calculate the sigma of the gamma1
                assign(term0, dgamma_dP0 * sigma_P0) # term0 = dgamma_dP0 * sigma_P0
                assign(term0, term0 * term0) # term0 = term0 * term0
                assign(term1, dgamma_dP1 * sigma_P1) # term1 = dgamma_dP1 * sigma_P1
                assign(term1, term1 * term1) # term1 = term1 * term1
                assign(term2, dgamma_dP3 * sigma_P3) # term2 = dgamma_dP3 * sigma_P3
                assign(term2, term2 * term2) # term2 = term2 * term2
                assign(sigma_gamma1, Math.sqrt(term0 + term1 + term2))
                save(sigma_gamma1, sigma_gamma1_st[i])

                # Save the time stamp of the measurement
                qubit.xy.play("x90", amplitude_scale=0, duration=4, timestamp_stream=f"time_stamp{i+1}")

                # Adaptive dt for the next repetition: dt ~ alpha * T1_est from gamma1_est
                if node.parameters.adaptive_dt:
                    assign(gamma_safe, GAMMA_ADAPTIVE_FLOOR + Math.relu(gamma1_est - GAMMA_ADAPTIVE_FLOOR))  # floor gamma1
                    assign(T1_est_scaled, Math.div(1.0, gamma_safe * TIME_SCALE_US))  # T1 in scaled fixed units
                    _delta_t_int_const = round(node.parameters.alpha * 250 * TIME_SCALE_US)  # alpha * T1 -> cycles
                    assign(delta_t_cycles, Cast.mul_int_by_fixed(_delta_t_int_const, T1_est_scaled))
                    with if_(delta_t_cycles < min_dt_cycles):
                        assign(delta_t_cycles, min_dt_cycles)  # clip to min_dt_ns
                    with if_(delta_t_cycles > max_dt_cycles):
                        assign(delta_t_cycles, max_dt_cycles)  # clip to max_dt_ns

                # Optional mid-run conventional T1 for comparison
                if measure_conventional_t1:
                    with if_(n == n_reps // 2):
                        qubit.xy.play(
                            "x90", amplitude_scale=0, duration=4, timestamp_stream=f"t1_conv_start{i + 1}"
                        )

                        with for_(t1_shot, 0, t1_shot < node.parameters.n_avg_per_point, t1_shot + 1):
                            with for_each_(t, t1_idle_times):
                                qubit.reset_qubit_active(
                                    pi_pulse_name="x180",
                                    readout_pulse_name="readout",
                                    max_attempts=node.parameters.reset_max_attempts,
                                )
                                qubit.align()
                                qubit.xy.play("x180")
                                qubit.wait(t)
                                qubit.align()
                                qubit.readout_state(state[i])
                                save(state[i], t1_state_st[i])

                        qubit.xy.play(
                            "x90", amplitude_scale=0, duration=4, timestamp_stream=f"t1_conv_end{i + 1}"
                        )

        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                gamma1_st[i].buffer(n_reps).save(f"gamma1{i + 1}")
                sigma_gamma1_st[i].buffer(n_reps).save(f"sigma_gamma1{i + 1}")
                dt_used_st[i].buffer(n_reps).save(f"dt_used{i + 1}")
                P0_st[i].buffer(n_reps).save(f"P0{i + 1}")
                P1_st[i].buffer(n_reps).save(f"P1{i + 1}")
                P3_st[i].buffer(n_reps).save(f"P3{i + 1}")
                P0_shots_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"P0_shots{i + 1}")
                P1_shots_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"P1_shots{i + 1}")
                P3_shots_st[i].buffer(n_reps, node.parameters.n_avg_per_point).save(f"P3_shots{i + 1}")
                if measure_conventional_t1:
                    t1_state_st[i].buffer(len(t1_idle_times)).average().save(f"t1_state{i + 1}")


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
        node.results["ds_raw"] = fetch_raw_dataset(
            job,
            qubits,
            n_reps,
            n_avg,
            node.namespace.get("t1_conventional_delays_ns")
            if node.parameters.measure_conventional_t1
            else None,
        )

# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["ade_clip_floors"] = ADE_CLIP_FLOORS


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw streams and extract per-qubit ADE fit results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results, node.results["time_to_decision_ms"] = fit_raw_data(
        node.results["ds_raw"], node
    )
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    ds = node.results["ds_fit"]
    qubits = node.namespace["qubits"]
    node.results["sigma_gamma1_fpga"] = {
        q.name: ds.sigma_gamma1.sel(qubit=q.name).values for q in qubits
    }
    node.results["T1_analytical_sigma_us"] = {k: v.sigma_T1_us for k, v in fit_results.items()}
    node.results["T1_clipped"] = {k: v.clipped for k, v in fit_results.items()}
    node.results["T1_bootstrap_sigma_us"] = {k: v.sigma_T1_boot_us for k, v in fit_results.items()}

    if "total_ms" in ds.attrs:
        node.results["t1_conventional_time_to_decision_ms"] = {
            k: float(ds.attrs[k])
            for k in ("measurement_ms", "fetch_ms", "analysis_ms", "total_ms")
            if k in ds.attrs
        }

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
    if "t1_conventional_decay" in figs:
        node.results["figures"]["t1_conventional_decay"] = figs["t1_conventional_decay"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
