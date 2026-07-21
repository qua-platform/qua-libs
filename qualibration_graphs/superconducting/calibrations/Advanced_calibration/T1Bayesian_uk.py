"""
        ADAPTIVE BAYESIAN T1 ESTIMATION  (u = 1/k reparametrization)

Real-time adaptive Bayesian estimation of qubit relaxation time T1, following
Berritta et al., Phys. Rev. X (2026), arXiv:2506.09576.

This is a variant of T1Bayesian.py that lets the posterior gamma SHAPE parameter
k grow far beyond the ~6.9 cap imposed by the QUA `fixed` [-8, 8) range, so the
reported credible interval (~ 1/sqrt(k)) can actually shrink.

Why the original caps k:
    The method-of-moments update (Eqs. 5-6) forms `k`, `k+1`, `k/theta` and
    `r^k = Math.pow(r, k)`. All of these need `k` (and `k+1`) to fit in `fixed`,
    which limits k < ~7. The paper's k reaches ~20-30, impossible to store in 4.28.

The u = 1/k trick (this file):
    We track the state as (T1, u) with u = 1/k. Since 1/k is always small it fits
    `fixed` with room to spare, so k is unbounded in storage. The update is
    rewritten so `k` is NEVER materialized:

      * shape update (Eq. 5b):   u_{i+1} = (1 + u) * (ratio_{k+1}/ratio_k) - 1
        (uses (k+1)/k = 1 + 1/k = 1 + u, so no reciprocal of u is ever taken)

      * T1 update (Eqs. 5-6):    T1_{i+1} = T1_i / ratio_k     (unchanged, stable)

      * the only quantity needing the exponent k is r^k. With adaptive
        tau = c*T1 we have r = theta/(theta+tau) = 1/(1 + c*u) and

            r^k = exp(k ln r) = exp(-c * phi(z)),   z = c*u,  phi(z) = ln(1+z)/z,

        which is an O(1) quantity even for large k. phi(z) is computed by a short
        Taylor series for small z (large k) and by Math.ln/Math.div for larger z
        (small k) so we never divide by a small number or exceed the fixed range.

    None of Math.pow(r, k), k+1, k/theta, or 1/u (with k>8) are ever evaluated,
    so k can grow to tens-hundreds. On-FPGA we store u; k = 1/u and theta = k*T1
    are reconstructed offline for plotting only.

Assumption: the Bayesian update models tau = c*T1 (the adaptive law). tau is still
clamped to [tau_min, tau_max] for the physical wait; in the normal regime the
clamp never triggers so the model is exact, and it only introduces a small
inconsistency in the pathological rails (T1 pinned at t1_min).

Prerequisites:
    - Calibrated readout and state discrimination (07_iq_blobs).
    - Calibrated x180 pulse (04b_power_rabi).
    - Working active reset.

State update:
    - None (tracking node; does not write qubit.T1).
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from qualibrate.core.parameters import RunnableParameters
from quam_config import Quam
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.parameters import get_qubits, QubitsExperimentNodeParameters, CommonNodeParameters
from qualibration_libs.analysis import fit_decay_exp, decay_exp
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qm.qua import *
from typing import Optional, List
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qualang_tools.results import progress_counter, fetching_tool
from scipy.signal import welch
from scipy.stats import norm, invgamma, gamma as gamma_dist

logger = logging.getLogger(__name__)

u = unit(coerce_to_integer=True)

# QUA `fixed` variables are in [-8, 8). Store times in ms internally; parameters/display in µs.
US_TO_MS = 1e-3
MS_TO_US = 1e3
MS_TO_CLK = 250_000.0  # ms -> ns (/1e6) -> 4 ns clock cycles (/4)
MS_TO_CLK_INT = 250_000  # int form for Cast.mul_int_by_fixed (avoids fixed overflow)

# %% {Helper functions}


def fetch_results_as_xarray(handles, qubits, measurement_axis, var_name):
    coords = {**measurement_axis, "qubit": [qubit.name for qubit in qubits]}
    coords = {key: coords[key] for key in reversed(coords.keys())}
    dim_names = list(coords.keys())
    per_qubit = [np.squeeze(handles.get(f"{var_name}{i + 1}").fetch_all()) for i in range(len(qubits))]
    values = np.stack(per_qubit, axis=0)
    while values.ndim > len(dim_names):
        values = np.squeeze(values, axis=-1)
    while values.ndim < len(dim_names):
        values = values[np.newaxis, ...]
    return xr.Dataset({var_name: (dim_names, values)}, coords=coords)


def fetch_t1_datasets(handles, qubit_list, n_reps, n_probes, interleaved, keep_shot_data):
    rep_axis = {"repetition": np.arange(n_reps)}
    probe_rep_axis = {"probe": np.arange(n_probes), "repetition": np.arange(n_reps)}

    ds_estimated_t1 = fetch_results_as_xarray(handles, qubit_list, rep_axis, "estimated_t1")
    ds_estimated_t1 = ds_estimated_t1.assign(estimated_t1=ds_estimated_t1.estimated_t1 * MS_TO_US)
    # The FPGA stores u = 1/k; reconstruct k and theta offline.
    ds_u = fetch_results_as_xarray(handles, qubit_list, rep_axis, "u_final")
    k_final = 1.0 / ds_u.u_final
    ds_k = xr.Dataset({"k_final": k_final}, coords=ds_u.coords)
    theta_final = ds_estimated_t1.estimated_t1 * k_final  # theta = k * T1 (µs)
    ds_theta = xr.Dataset({"theta_final": theta_final}, coords=ds_estimated_t1.coords)
    ds_time_stamp = fetch_results_as_xarray(handles, qubit_list, rep_axis, "time_stamp")
    timestamp_values = ds_time_stamp.time_stamp.values
    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim="repetition")) * 4e-9).time_stamp

    parts = [ds_estimated_t1, ds_k, ds_theta, time_stamp]
    ds_state = ds_tau = ds_state_lin = None
    if keep_shot_data:
        ds_state = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "state")
        ds_tau = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "tau_ms")
        ds_tau = ds_tau.assign(tau_us=ds_tau.tau_ms * MS_TO_US).drop_vars("tau_ms")
        parts.extend([ds_state, ds_tau])
    if interleaved:
        ds_state_lin = fetch_results_as_xarray(handles, qubit_list, probe_rep_axis, "state_lin")
        parts.append(ds_state_lin)
    return xr.merge(parts), time_stamp, ds_state, ds_tau, ds_state_lin


def welch_segment_params(n_samples, max_nperseg=1024):
    """Return (nperseg, noverlap) valid for scipy.signal.welch with n_samples points."""
    if n_samples < 4:
        return None
    nperseg = min(max_nperseg, n_samples)
    if nperseg < 8:
        nperseg = n_samples
    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    return nperseg, noverlap


def posterior_t1_credible_interval(k, t1_est_us, ci=0.9):
    """Approximate credible interval on T1 from posterior shape k (T1 mean = t1_est_us)."""
    k = np.asarray(k, dtype=float)
    t1_est_us = np.asarray(t1_est_us, dtype=float)
    # Posterior relative uncertainty scales roughly as 1/sqrt(k) for the gamma model.
    t1_std = t1_est_us / np.sqrt(np.maximum(k, 1.0))
    z = float(norm.ppf(0.5 + ci / 2))
    t1_ci_low = np.maximum(t1_est_us - z * t1_std, 0.0)
    t1_ci_high = t1_est_us + z * t1_std
    return t1_est_us, t1_ci_low, t1_ci_high


def resolve_t1_prior_us_per_qubit(node, qubit_list):
    """Per-qubit prior mean T1 in µs (from QuAM qubit.T1 when available)."""
    fallback = float(node.parameters.t1_prior_us)
    priors = []
    for q in qubit_list:
        t1_s = getattr(q, "T1", None) if node.parameters.use_quam_t1_prior else None
        priors.append(float(t1_s) * 1e6 if t1_s is not None and t1_s > 0 else fallback)
    return priors


def overlapping_allan_deviation(trace, dt, max_points=80):
    """Allan deviation of a T1 time trace (paper Eq. 8, discrete approximation)."""
    n = len(trace)
    if n < 4:
        return np.array([]), np.array([])
    max_m = min(n // 2, max_points)
    if max_m < 2:
        return np.array([]), np.array([])
    ms = np.unique(np.geomspace(1, n // 2, num=max_m, dtype=int))
    taus, adevs = [], []
    for m in ms:
        tau = m * dt
        means = np.convolve(trace, np.ones(m) / m, mode="valid")
        if len(means) <= m:
            continue
        deltas = means[m:] - means[:-m]
        adevs.append(np.sqrt(0.5 * np.mean(deltas**2)))
        taus.append(tau)
    return np.array(taus), np.array(adevs)


# NOTE: Bayesian update must be inlined in the QUA program. Passing k[i] into a
# Python helper does not reliably update the underlying QUA variable by reference.


# %% {Node_parameters}
class NodeSpecificParameters(RunnableParameters):
    """Qubits to calibrate."""

    num_repetitions: int = 100
    """Number of independent T1 estimation blocks (time-trace length)."""

    num_probes: int = 100
    """Number of adaptive single-shot probes per estimation block."""

    c: float = 0.51
    """Adaptive waiting-time coefficient: tau = c * T1_est. Paper's theoretically
    motivated optimum is c ~ 0.51 (valid range c in (0, 1.59))."""

    k0: float = 1.0
    """Initial gamma shape parameter for Gamma_1 prior."""

    t1_prior_us: float = 35.0
    """Prior mean T1 in µs (internal theta0_ms = k0 * t1_prior_us * 1e-3)."""

    use_quam_t1_prior: bool = False
    """If True and qubit.T1 is set in QuAM, use it instead of t1_prior_us."""

    t1_min_us: float = 1.0
    """Lower bound on adaptive T1 estimate (µs), applied on FPGA after each update."""

    t1_max_us: float = 100.0
    """Upper bound on adaptive T1 estimate (µs), applied on FPGA after each update."""

    interleaved_validation: bool = True
    """If True, add a non-adaptive T1 shot after each adaptive probe (paper Fig. 2c)."""

    min_wait_time_in_ns: int = 16
    """Minimum linear wait time for interleaved validation grid."""

    max_wait_time_in_ns: int = 200_000
    """Maximum linear wait time for interleaved validation grid."""

    keep_shot_data: bool = True
    """Stream per-probe outcomes and adaptive waiting times."""

    credible_interval: float = 0.90
    """Credible interval level for T1 uncertainty band in plots."""

    k_max: float = 100.0
    """Upper cap on posterior gamma shape k. Thanks to the u = 1/k parametrization
    this can be much larger than the original ~6.9 cap: k itself is never stored,
    only u = 1/k. Keep it below a few hundred (u ~ 1/k must stay well above the
    fixed resolution). Larger k -> tighter credible interval (~1/sqrt(k))."""

    k_min: float = 0.2
    """Lower bound on posterior gamma shape k. Bounded so that (1 + c*(1/k_min))
    stays inside the fixed range for the ln/inv used in r^k."""

    active_reset_per_probe: bool = False
    """If True, active reset also runs before each interleaved validation shot."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    NodeSpecificParameters,
):
    qubits: Optional[List[str]] = ["qA5"]


node = QualibrationNode(name="T1Bayesian_uk", parameters=Parameters(), machine=Quam.load())

# %% {Initialize_QuAM_and_QOP}
machine = node.machine

if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = get_qubits(node)
qubit_list = list(qubits)
num_qubits = len(qubit_list)

n_reps = node.parameters.num_repetitions
n_probes = node.parameters.num_probes
k0 = float(node.parameters.k0)
t1_prior_us_list = resolve_t1_prior_us_per_qubit(node, qubit_list)
t1_prior_ms_list = [t1_us * US_TO_MS for t1_us in t1_prior_us_list]
t1_prior_by_name = dict(zip([q.name for q in qubit_list], t1_prior_us_list))
t1_min_ms = float(node.parameters.t1_min_us) * US_TO_MS
t1_max_ms = float(node.parameters.t1_max_us) * US_TO_MS
tau_min_ms = 0.001  # 1 µs minimum adaptive wait
tau_max_ms = t1_max_ms
c_adaptive = float(node.parameters.c)
interleaved = node.parameters.interleaved_validation
active_reset_per_probe = node.parameters.active_reset_per_probe

# We store u = 1/k (always small) instead of k, so k is unbounded in storage.
# Bound k in [k_min, k_max] by clamping u in [1/k_max, 1/k_min]. The only upper
# limit on k comes from fixed precision (u must stay well above ~1e-6). The only
# lower limit on k is that 1 + c*u_max must stay inside the fixed range because
# it feeds Math.ln / Math.inv when computing r^k.
u_max_safe = (7.5 - 1.0) / c_adaptive  # keep 1 + c*u < 7.5 < 8
k_min_val = max(float(node.parameters.k_min), 1.0 / u_max_safe)
k_max_val = float(node.parameters.k_max)
u_min = 1.0 / k_max_val
u_max = 1.0 / k_min_val
u0 = 1.0 / k0

lin_times_clocks = np.linspace(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    n_probes,
    dtype=int,
)
lin_times_clocks = np.maximum(lin_times_clocks, 4)

# %% {QUA_program}
with program() as T1Bayes:
    I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
    rep = declare(int)
    probe = declare(int)

    state = [declare(int) for _ in range(num_qubits)]
    state_lin = [declare(int) for _ in range(num_qubits)]

    # Posterior state is tracked as (t1_est, u) with u = 1/k. k is NEVER stored.
    u_inv_k = [declare(fixed) for _ in range(num_qubits)]  # u = 1/k
    t1_est = [declare(fixed) for _ in range(num_qubits)]  # T1 estimate (ms)
    alpha = [declare(fixed) for _ in range(num_qubits)]
    beta = [declare(fixed) for _ in range(num_qubits)]

    tau_ms = [declare(fixed) for _ in range(num_qubits)]
    tau_clocks = [declare(int) for _ in range(num_qubits)]

    z = declare(fixed)  # z = tau/theta = c*u   (O(1), always in fixed range)
    z2 = declare(fixed)
    phi = declare(fixed)  # phi(z) = ln(1+z)/z
    lexp = declare(fixed)  # L = k*ln(r) = -c*phi(z)
    r = declare(fixed)
    r_pow_k = declare(fixed)
    r_pow_k1 = declare(fixed)
    r_pow_k2 = declare(fixed)
    spam = declare(fixed)
    num_k = declare(fixed)
    den_k = declare(fixed)
    num_k1 = declare(fixed)
    den_k1 = declare(fixed)
    ratio_k = declare(fixed)
    ratio_k1 = declare(fixed)
    ratio_ratio = declare(fixed)  # R = ratio_k1 / ratio_k

    lin_tau = declare(int, value=lin_times_clocks.tolist())

    estimated_t1_st = [declare_stream() for _ in range(num_qubits)]
    u_final_st = [declare_stream() for _ in range(num_qubits)]

    # Per-probe posterior (u, T1) of the LAST repetition, to visualize convergence.
    u_evol_st = [declare_stream() for _ in range(num_qubits)]
    t1_evol_st = [declare_stream() for _ in range(num_qubits)]

    state_st = [declare_stream() for _ in range(num_qubits)]
    tau_st = [declare_stream() for _ in range(num_qubits)]
    state_lin_st = [declare_stream() for _ in range(num_qubits)]

    for multiplexed_qubits in qubits.batch():
        for i, qubit in multiplexed_qubits.items():
            # Paper Eq. (1): alpha = P(|0> read |1> true), beta = P(|1> read |0> true).
            # Confusion matrix rows are prep |g>, |e>; cols are meas |g>, |e>.
            assign(alpha[i], qubit.resonator.confusion_matrix[1][0])  # eg
            assign(beta[i], qubit.resonator.confusion_matrix[0][1])  # ge

        for qubit in multiplexed_qubits.values():
            machine.initialize_qpu(target=qubit)

        with for_(rep, 0, rep < n_reps, rep + 1):
            save(rep, n_st)

            for i, qubit in multiplexed_qubits.items():
                assign(u_inv_k[i], u0)
                assign(t1_est[i], t1_prior_ms_list[i])
                assign(state[i], 0)

            for i, qubit in multiplexed_qubits.items():
                qubit.reset("active", False)
            align()

            with for_(probe, 0, probe < n_probes, probe + 1):
                # Eq. (4): tau_i = c * T1_est from the current prior (u_i, T1_i).
                for i, qubit in multiplexed_qubits.items():
                    # Record the posterior (u, T1) entering this probe of the last
                    # repetition, plus a hardware timestamp for the execution-time axis.
                    with if_(rep == n_reps - 1):
                        save(u_inv_k[i], u_evol_st[i])
                        save(t1_est[i], t1_evol_st[i])
                        qubit.xy.play(
                            "x180",
                            amplitude_scale=0,
                            duration=4,
                            timestamp_stream=f"probe_time_stamp{i + 1}",
                        )
                    assign(tau_ms[i], c_adaptive * t1_est[i])
                    with if_(tau_ms[i] < tau_min_ms):
                        assign(tau_ms[i], tau_min_ms)
                    with if_(tau_ms[i] > tau_max_ms):
                        assign(tau_ms[i], tau_max_ms)
                    # ms -> 4 ns clock cycles. Use int*fixed: `tau_ms * 250000` in fixed
                    # arithmetic would exceed 8 and wrap (modulo 16) to a garbage wait.
                    assign(tau_clocks[i], Cast.mul_int_by_fixed(MS_TO_CLK_INT, tau_ms[i]))
                    with if_(tau_clocks[i] < 4):
                        assign(tau_clocks[i], 4)

                # Prep |e>, wait, measure, then flip back to |g> when outcome is |1>.
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.resonator.wait(tau_clocks[i])
                    qubit.readout_state(state[i])
                    qubit.align()
                    qubit.xy.play("x180", condition=Cast.to_bool(state[i]))
                    if node.parameters.keep_shot_data:
                        save(state[i], state_st[i])
                        save(tau_ms[i], tau_st[i])
                align()

                for i, qubit in multiplexed_qubits.items():
                    # Timestamp the math (Bayesian update) block on the last repetition so we
                    # can measure how long one update takes. The diff of these two markers
                    # includes the leading 4-cycle marker pulse, subtracted off in analysis.
                    with if_(rep == n_reps - 1):
                        qubit.xy.play(
                            "x180", amplitude_scale=0, duration=4,
                            timestamp_stream=f"math_start{i + 1}",
                        )
                    # Method-of-moments gamma update (Berritta et al. Eqs. 5-6) in the
                    # u = 1/k parametrization, so k is NEVER materialized (can grow large).
                    #
                    #   ratio_k   = [num @ k]   / [den @ k]      (Eq. 6 fraction)
                    #   ratio_k1  = [num @ k+1] / [den @ k+1]
                    #   T1_{i+1}  = T1_i / ratio_k                         (Eqs. 5-6)
                    #   u_{i+1}   = (1 + u) * ratio_k1/ratio_k - 1         (Eq. 5b, (k+1)/k = 1+u)
                    #
                    # r^k is the only place the exponent k appears. With tau = c*T1,
                    #   r = theta/(theta+tau) = 1/(1 + c*u),  z = c*u,
                    #   r^k = exp(k ln r) = exp(-c * phi(z)),  phi(z) = ln(1+z)/z,
                    # an O(1) quantity for any k. phi is a Taylor series for small z
                    # (large k) and ln/div for larger z (small k) so nothing overflows
                    # and we never divide by a small number.
                    assign(spam, 1.0 - alpha[i] - beta[i])
                    assign(z, c_adaptive * u_inv_k[i])  # z = tau/theta = c*u
                    assign(r, Math.inv(1.0 + z))  # r = 1/(1+z), 1+z >= 1 (safe for inv)
                    with if_(z < 0.2):
                        # phi(z) = 1 - z/2 + z^2/3 - z^3/4 (err < 1e-4 for z<0.2). No division.
                        assign(z2, z * z)
                        assign(phi, 1.0 - 0.5 * z + z2 * (1.0 / 3.0) - (z2 * z) * (1.0 / 4.0))
                    with else_():
                        # z >= 0.2 > 1/8, so Math.div is safe here.
                        assign(phi, Math.div(Math.ln(1.0 + z), z))
                    assign(lexp, -c_adaptive * phi)  # L = k ln r
                    assign(r_pow_k, Math.exp(lexp))  # r^k  (~exp(-c), O(1))
                    assign(r_pow_k1, r_pow_k * r)  # r^(k+1)
                    assign(r_pow_k2, r_pow_k1 * r)  # r^(k+2)

                    with if_(state[i] == 0):
                        # Eq. (6a): f_0 numerator/denominator at k and k+1.
                        assign(num_k, 1.0 - beta[i] - spam * r_pow_k1)
                        assign(den_k, 1.0 - beta[i] - spam * r_pow_k)
                        assign(num_k1, 1.0 - beta[i] - spam * r_pow_k2)
                        assign(den_k1, 1.0 - beta[i] - spam * r_pow_k1)
                    with else_():
                        # Eq. (6b): f_1 numerator/denominator at k and k+1.
                        assign(num_k, beta[i] + spam * r_pow_k1)
                        assign(den_k, beta[i] + spam * r_pow_k)
                        assign(num_k1, beta[i] + spam * r_pow_k2)
                        assign(den_k1, beta[i] + spam * r_pow_k1)

                    assign(ratio_k, Math.div(num_k, den_k))  # O(1)
                    assign(ratio_k1, Math.div(num_k1, den_k1))  # O(1)

                    # T1_{i+1} = 1/f_m(k) = T1_i / ratio_k  (Eqs. 5-6 combined).
                    assign(t1_est[i], Math.div(t1_est[i], ratio_k))
                    with if_(t1_est[i] < t1_min_ms):
                        assign(t1_est[i], t1_min_ms)
                    with if_(t1_est[i] > t1_max_ms):
                        assign(t1_est[i], t1_max_ms)

                    # Eq. (5b): u_{i+1} = 1/k_{i+1} = (1 + u) * ratio_k1/ratio_k - 1.
                    # Reciprocal-free: no 1/u and no k are ever formed, so k = 1/u can
                    # grow well past 8 (only u is stored). Clamp u to [1/k_max, 1/k_min].
                    assign(ratio_ratio, Math.div(ratio_k1, ratio_k))
                    assign(u_inv_k[i], (1.0 + u_inv_k[i]) * ratio_ratio - 1.0)
                    with if_(u_inv_k[i] < u_min):
                        assign(u_inv_k[i], u_min)
                    with if_(u_inv_k[i] > u_max):
                        assign(u_inv_k[i], u_max)

                    # End-of-math timestamp (last repetition only).
                    with if_(rep == n_reps - 1):
                        qubit.xy.play(
                            "x180", amplitude_scale=0, duration=4,
                            timestamp_stream=f"math_end{i + 1}",
                        )

                if interleaved:
                    # The adaptive probe's conditional x180 (above) already left the qubit
                    # in |g>. Do NOT re-flip on `state` here: doing so would prepare |g>
                    # (not |e>) whenever the adaptive outcome was m=1, biasing the whole
                    # non-adaptive decay curve. Optionally run a full active reset instead.
                    if active_reset_per_probe:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset("active", False)
                        align()

                    # Non-adaptive shot: prepare |e>, then wait the linear grid time.
                    for i, qubit in multiplexed_qubits.items():
                        assign(tau_clocks[i], lin_tau[probe])
                        qubit.xy.play("x180")
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(tau_clocks[i])
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.readout_state(state_lin[i])
                        save(state_lin[i], state_lin_st[i])
                        with if_(state_lin[i] == 1):
                            qubit.xy.play("x180")
                    align()

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180", amplitude_scale=0, duration=4, timestamp_stream=f"time_stamp{i + 1}")
                save(t1_est[i], estimated_t1_st[i])
                save(u_inv_k[i], u_final_st[i])

    align()
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            estimated_t1_st[i].buffer(n_reps).save(f"estimated_t1{i + 1}")
            u_final_st[i].buffer(n_reps).save(f"u_final{i + 1}")
            u_evol_st[i].buffer(n_probes).save(f"u_evol{i + 1}")
            t1_evol_st[i].buffer(n_probes).save(f"t1_evol{i + 1}")
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps, n_probes).save(f"state{i + 1}")
                tau_st[i].buffer(n_reps, n_probes).save(f"tau_ms{i + 1}")
            if interleaved:
                state_lin_st[i].buffer(n_reps, n_probes).save(f"state_lin{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    pass
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config=machine.generate_config(), timeout=node.parameters.timeout) as qm:
        job = qm.execute(T1Bayes)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n_val = results.fetch_all()[0]
            progress_counter(n_val, n_reps, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    ds, time_stamp, ds_state, ds_tau, ds_state_lin = fetch_t1_datasets(
        job.result_handles,
        qubit_list,
        n_reps,
        n_probes,
        interleaved,
        node.parameters.keep_shot_data,
    )
    node.results = {"ds": ds}
    ds_estimated_t1 = ds
    ds_k = ds
    ds_theta = ds

    # Per-probe posterior evolution of the last repetition (probe axis only).
    probe_axis = {"probe": np.arange(n_probes)}
    ds_u_evol = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "u_evol")
    ds_k_evol = xr.Dataset({"k_evol": 1.0 / ds_u_evol.u_evol}, coords=ds_u_evol.coords)
    ds_t1_evol = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "t1_evol")
    ds_t1_evol = ds_t1_evol.assign(t1_evol=ds_t1_evol.t1_evol * MS_TO_US)  # ms -> us
    # Per-probe hardware timestamps (clock cycles) -> elapsed execution time (ms) within the round.
    ds_probe_ts = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "probe_time_stamp")
    exec_time_ms = (ds_probe_ts.probe_time_stamp - ds_probe_ts.probe_time_stamp.isel(probe=0)) * 4e-9 * 1e3

    # Duration of the math (Bayesian update) block per probe (last repetition).
    # 4 ns per clock cycle; subtract the 4-cycle leading marker pulse. 1e3 -> µs.
    ds_math_start = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "math_start")
    ds_math_end = fetch_results_as_xarray(job.result_handles, qubit_list, probe_axis, "math_end")
    math_time_us = (ds_math_end.math_end - ds_math_start.math_start - 4) * 4e-9 * 1e6
    avg_math_time_us = {q: float(math_time_us.sel(qubit=q).mean()) for q in ds_math_start.qubit.values}
    for q, t_us in avg_math_time_us.items():
        logger.info("[%s] average Bayesian-update (math) time = %.3f µs", q, t_us)

    # %% {Data_analysis}
    ci = node.parameters.credible_interval
    t1_est = ds_estimated_t1.estimated_t1.values
    k_vals = ds_k.k_final.values
    theta_vals = ds_theta.theta_final.values
    n_qubits_plot, _ = t1_est.shape

    _, t1_ci_low, t1_ci_high = posterior_t1_credible_interval(k_vals, t1_est, ci=ci)

    ds_ci = xr.Dataset(
        {
            "t1_ci_low": (("qubit", "repetition"), t1_ci_low),
            "t1_ci_high": (("qubit", "repetition"), t1_ci_high),
        },
        coords={"qubit": ds_estimated_t1.qubit.values, "repetition": np.arange(n_reps)},
    )

    t_vals = time_stamp.values
    dt = np.mean(np.diff(t_vals, axis=1))
    fs = 1.0 / dt if dt > 0 else 1.0
    welch_params = welch_segment_params(n_reps)

    ds_welch = None
    if welch_params is not None:
        nperseg, noverlap = welch_params
        welch_freqs_list = []
        welch_psd_list = []
        for qidx in range(n_qubits_plot):
            y = t1_est[qidx] - np.nanmean(t1_est[qidx])
            f_welch, pxx = welch(
                y,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                detrend="constant",
            )
            welch_freqs_list.append(f_welch)
            welch_psd_list.append(pxx)

        welch_freqs = welch_freqs_list[0]
        welch_psd_arr = np.stack(welch_psd_list, axis=0)
        ds_welch = xr.DataArray(
            welch_psd_arr,
            dims=("qubit", "frequency"),
            coords={"qubit": ds_estimated_t1.qubit.values, "frequency": welch_freqs},
            name="t1_welch_psd",
        )

    allan_tau_list = []
    allan_dev_list = []
    for qidx in range(n_qubits_plot):
        taus_a, adev_a = overlapping_allan_deviation(t1_est[qidx], dt)
        allan_tau_list.append(taus_a)
        allan_dev_list.append(adev_a)

    max_len = max(len(a) for a in allan_tau_list) if allan_tau_list else 0
    if max_len > 0:
        allan_tau_arr = np.full((n_qubits_plot, max_len), np.nan)
        allan_dev_arr = np.full((n_qubits_plot, max_len), np.nan)
        for qidx in range(n_qubits_plot):
            n_a = len(allan_tau_list[qidx])
            allan_tau_arr[qidx, :n_a] = allan_tau_list[qidx]
            allan_dev_arr[qidx, :n_a] = allan_dev_list[qidx]
        ds_allan = xr.Dataset(
            {
                "allan_tau": (("qubit", "allan_index"), allan_tau_arr),
                "allan_deviation": (("qubit", "allan_index"), allan_dev_arr),
            },
            coords={"qubit": ds_estimated_t1.qubit.values, "allan_index": np.arange(max_len)},
        )
    else:
        ds_allan = None

    # %% {Plotting}
    grid_t1 = QubitGrid(ds, [q.grid_location for q in qubit_list])
    for ax, qubit in grid_iter(grid_t1):
        qname = qubit["qubit"]
        t_axis = time_stamp.sel(qubit=qname).values
        y = ds_estimated_t1.estimated_t1.sel(qubit=qname).values
        lo = ds_ci.t1_ci_low.sel(qubit=qname).values
        hi = ds_ci.t1_ci_high.sel(qubit=qname).values
        # Mean wall-clock time to produce one estimate (one repetition of n_probes probes).
        dt_est = np.diff(t_axis)
        est_time_ms = float(np.mean(dt_est)) * 1e3 if dt_est.size else np.nan
        ax.plot(t_axis, y, "o-", alpha=0.6, markersize=3, label="T1 estimate")
        ax.fill_between(t_axis, lo, hi, alpha=0.25, label=f"{int(ci * 100)}% CI")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("T1 (µs)")
        ax.set_title(f"{qname}\nestimation time = {est_time_ms:.3f} ms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    grid_t1.fig.suptitle("Adaptive Bayesian T1 trace (u = 1/k)")
    plt.tight_layout()
    node.results["t1_trace_figure"] = grid_t1.fig

    # {Posterior P(T1) evolution vs execution time (last repetition)}
    # Posterior over the relaxation rate Gamma1 is Gamma(shape=k, rate=theta) with
    # theta = k * T1 (Berritta et al.). The posterior over T1 = 1/Gamma1 is therefore
    # inverse-gamma(shape=k, scale=theta). We visualize how it sharpens probe-by-probe
    # during the last repetition to check convergence.
    t1_grid_us = np.linspace(
        max(1.0, 0.5 * float(node.parameters.t1_min_us)),
        1.2 * float(node.parameters.t1_max_us),
        400,
    )
    grid_conv = QubitGrid(ds_t1_evol, [q.grid_location for q in qubit_list])
    for ax, qubit in grid_iter(grid_conv):
        qname = qubit["qubit"]
        k_ev = ds_k_evol.k_evol.sel(qubit=qname).values  # (probe,)
        t1_ev_us = ds_t1_evol.t1_evol.sel(qubit=qname).values  # (probe,) in µs
        t_exec_ms = exec_time_ms.sel(qubit=qname).values  # (probe,) elapsed time (ms)
        theta_us = k_ev * t1_ev_us  # scale of inverse-gamma
        pdf = np.zeros((t_exec_ms.size, t1_grid_us.size))
        for p in range(t_exec_ms.size):
            pdf[p] = invgamma.pdf(t1_grid_us, a=k_ev[p], scale=theta_us[p])
        pcm = ax.pcolormesh(t1_grid_us, t_exec_ms, pdf, cmap="viridis", shading="auto")
        ax.plot(t1_ev_us, t_exec_ms, "w.-", lw=1, ms=3, label="T1 estimate")
        ax.set_xlabel("T1 (µs)")
        ax.set_ylabel("execution time (ms)")
        ax.set_title(f"{qname}\nfinal T1={t1_ev_us[-1]:.1f} µs, k={k_ev[-1]:.2f}")
        ax.legend(fontsize=7, loc="upper right")
        grid_conv.fig.colorbar(pcm, ax=ax, label="P(T1)")
    grid_conv.fig.suptitle("Posterior P(T1) evolution vs execution time (last repetition)")
    plt.tight_layout()
    node.results["gamma_evolution_figure"] = grid_conv.fig

    # {Gamma posterior P(Γ₁) evolution within the last estimation round}
    # The posterior over the relaxation rate Gamma1 is Gamma(shape=k, rate=theta=k*T1).
    # Render the actual gamma pdf as a 2D colormap: Γ₁ on x, the Bayesian update step
    # (one per probe within the round) on y, color = P(Γ₁), to see the distribution
    # sharpen with each successive measurement/update.
    g1_max = 1.0 / node.parameters.t1_min_us
    g1_grid = np.linspace(1e-4, g1_max, 500)
    step_idx = ds_t1_evol.probe.values
    grid_gam = QubitGrid(ds_t1_evol, [q.grid_location for q in qubit_list])
    for ax, qubit in grid_iter(grid_gam):
        qname = qubit["qubit"]
        k_ev = ds_k_evol.k_evol.sel(qubit=qname).values
        t1_ev_us = ds_t1_evol.t1_evol.sel(qubit=qname).values
        theta_us = k_ev * t1_ev_us  # rate of the gamma (µs)
        pdf_g = np.zeros((step_idx.size, g1_grid.size))
        for p in range(step_idx.size):
            pdf_g[p] = gamma_dist.pdf(g1_grid, a=k_ev[p], scale=1.0 / theta_us[p])
        pcm = ax.pcolormesh(g1_grid, step_idx, pdf_g, cmap="viridis", shading="auto")
        ax.plot(1.0 / t1_ev_us, step_idx, "w.-", lw=1, ms=3, label="Γ₁ estimate")
        ax.set_xlabel("Γ₁ = 1/T1 (1/µs)")
        ax.set_ylabel("Bayesian update step")
        ax.set_title(f"{qname}\nfinal T1={t1_ev_us[-1]:.1f} µs, k={k_ev[-1]:.2f}")
        ax.legend(fontsize=7, loc="upper right")
        grid_gam.fig.colorbar(pcm, ax=ax, label="P(Γ₁)")
    grid_gam.fig.suptitle("Gamma posterior P(Γ₁) evolution within last estimation round")
    plt.tight_layout()
    node.results["gamma_pdf_evolution_figure"] = grid_gam.fig

    if ds_welch is not None:
        grid_welch = QubitGrid(ds_welch.to_dataset(name="t1_welch_psd"), [q.grid_location for q in qubit_list])
        for ax, qubit in grid_iter(grid_welch):
            qname = qubit["qubit"]
            ds_welch.sel(qubit=qname).plot(ax=ax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Welch PSD of T1")
            ax.set_title(qname)
        grid_welch.fig.suptitle("T1 fluctuation Welch PSD")
        plt.tight_layout()
        node.results["welch_psd_figure"] = grid_welch.fig

    if ds_allan is not None:
        grid_allan = QubitGrid(ds_allan, [q.grid_location for q in qubit_list])
        for ax, qubit in grid_iter(grid_allan):
            qname = qubit["qubit"]
            tau_a = ds_allan.allan_tau.sel(qubit=qname).values
            adev = ds_allan.allan_deviation.sel(qubit=qname).values
            mask = np.isfinite(tau_a) & np.isfinite(adev) & (tau_a > 0) & (adev > 0)
            ax.loglog(tau_a[mask], adev[mask], "o-")
            ax.set_xlabel("Averaging time τ (s)")
            ax.set_ylabel("Allan deviation σ(T1)")
            ax.set_title(qname)
            ax.grid(True, which="both", alpha=0.3)
        grid_allan.fig.suptitle("T1 Allan deviation")
        plt.tight_layout()
        node.results["allan_deviation_figure"] = grid_allan.fig

    if node.parameters.keep_shot_data:
        grid_shot = QubitGrid(ds, [q.grid_location for q in qubit_list])
        for ax, qubit in grid_iter(grid_shot):
            qname = qubit["qubit"]
            tau_vals = np.maximum(ds_tau.tau_us.sel(qubit=qname).values, 0.0)
            states = ds_state.state.sel(qubit=qname).values
            t_axis = time_stamp.sel(qubit=qname).values
            Y = np.repeat(t_axis[:, None], states.shape[1], axis=1)
            X = tau_vals
            pcm = ax.pcolormesh(X, Y, states, cmap="coolwarm", vmin=0, vmax=1, shading="nearest")
            ax.set_xlabel("adaptive τ (µs)")
            ax.set_ylabel("time (s)")
            ax.set_title(qname)
            grid_shot.fig.colorbar(pcm, ax=ax, label="state")
        grid_shot.fig.suptitle("Single-shot outcomes vs adaptive wait time")
        plt.tight_layout()
        node.results["shot_data_figure"] = grid_shot.fig

    if interleaved:
        lin_times_ns = 4 * lin_times_clocks
        p_exc_lin = (
            ds_state_lin.state_lin.mean(dim="repetition")
            .rename(probe="idle_time")
            .assign_coords(idle_time=("idle_time", lin_times_ns))
        )
        fit_data = fit_decay_exp(p_exc_lin, "idle_time")
        tau_fit = -1 / fit_data.sel(fit_vals="decay")

        grid_val = QubitGrid(ds_state_lin, [q.grid_location for q in qubit_list])
        for ax, qubit in grid_iter(grid_val):
            qname = qubit["qubit"]
            p = p_exc_lin.sel(qubit=qname).values
            t1_adaptive = float(ds_estimated_t1.estimated_t1.sel(qubit=qname).mean())
            t1_fit = float(tau_fit.sel(qubit=qname).values) * 1e-3
            ax.plot(lin_times_ns * 1e-3, p, "o-", label="interleaved non-adaptive")
            t_fit = np.linspace(lin_times_ns.min(), lin_times_ns.max(), 200)
            ax.plot(
                t_fit * 1e-3,
                decay_exp(
                    t_fit,
                    fit_data.sel(qubit=qname, fit_vals="a").values,
                    fit_data.sel(qubit=qname, fit_vals="offset").values,
                    fit_data.sel(qubit=qname, fit_vals="decay").values,
                ),
                "--",
                color="gray",
                label=f"exp. fit T1={t1_fit:.1f} µs",
            )
            ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)
            ax.set_xlabel("τ_lin (µs)")
            ax.set_ylabel("P(|1⟩)")
            ax.set_title(f"{qname}\nadaptive mean T1={t1_adaptive:.1f} µs")
            ax.legend(fontsize=8)
        grid_val.fig.suptitle("Interleaved validation: non-adaptive vs adaptive")
        plt.tight_layout()
        node.results["validation_figure"] = grid_val.fig
        node.results["validation_t1_fit_us"] = {
            q: float(tau_fit.sel(qubit=q).values) * 1e-3 for q in ds_estimated_t1.qubit.values
        }
        for q in ds_estimated_t1.qubit.values:
            t1_adaptive = float(ds_estimated_t1.estimated_t1.sel(qubit=q).mean())
            t1_na = float(node.results["validation_t1_fit_us"][q])
            if t1_na > 0 and (t1_adaptive > 3 * t1_na or t1_adaptive < t1_na / 3):
                logger.warning(
                    "[%s] adaptive mean T1=%.1f µs disagrees with interleaved fit T1=%.1f µs. "
                    "Check t1_prior_us (used %.1f µs) and re-run with t1_prior_us≈%.0f or calibrate qubit.T1 in QuAM.",
                    q,
                    t1_adaptive,
                    t1_na,
                    t1_prior_by_name[q],
                    t1_na,
                )

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.results["t1_prior_us_used"] = t1_prior_by_name
    node.results["avg_math_time_us"] = avg_math_time_us
    if ds_welch is not None:
        node.results["ds_welch"] = ds_welch
    if ds_allan is not None:
        node.results["ds_allan"] = ds_allan
    node.machine = machine
    node.save()
# %%
