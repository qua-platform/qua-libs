"""Analysis test for 17_geometric_cz_error_amplification.

Uses virtual_qpu (Loss-DiVincenzo Hamiltonian + dynamiqs) end-to-end:

1. Run the same fixed-amplitude duration sweep as node 16a at a chosen exchange
   voltage (0.25 V) to find the duration where summed cphase is π.
2. Snap that duration to the nearest point on the 16a grid (4 ns steps; closest
   to the fitted π-time).
3. At that snapped duration, run a node-16b-style amplitude sweep to find the
   exchange amplitude where the conditional phase reaches π (= amplitude center
   for error amplification).
4. At that duration, repeat the raw CPhase block N times (multiples of 2:
   2, 4, …, max) with full two-qubit physics, two control states, and the
   same time ordering as the QUA program.  Batch axes use
   ``virtual_qpu.sweep`` (nested ``jax.vmap``) over
   ``(n_gates, exchange_amplitude)`` with a fixed padded cphase stack.

I/Q data for the analysis uses target Bloch components before the readout
π/2: ``E = 0.5 * (1 + ⟨σ_α⟩)`` for α∈{x,y} in the drive frame, so the
conditional-parity I/Q phasor matches the phasor fit (same as a parity-like
0.5-centered Ramsey in I/Q, without mixing in Z readout of the wrong
quadrature).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import dynamiqs as dq
import jax.numpy as jnp

from calibration_utils.geometric_cz_amplitude.analysis import (
    fit_raw_data as fit_amplitude_raw_data,
)
from calibration_utils.geometric_cz_duration.analysis import (
    fit_raw_data as fit_duration_raw_data,
)

from virtual_qpu.dynamics import simulate as _vqpu_simulate
from virtual_qpu.operators import expval as _vqpu_expval
from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as _vqpu_sweep

from .conftest import (
    ARTIFACTS_BASE,
    DEFAULT_PULSE_DURATION_NS,
    QUBIT_PAIR_NAMES,
    build_joint_stream_analysis_ds,
    simulate_sweep,
)

NODE_NAME = "17_geometric_cz_error_amplification"

# Fixed exchange bias for node-16a-style π-time search and for the amplification sweep.
EXCHANGE_AMPLITUDE_REF_V = 0.25

# 16a-style duration grid (must match analysis test 16a for comparable fit).
DURATION_STEP_NS = 4.0
DUR_MIN = 16.0
DUR_MAX = 420.0
N_DURATIONS = int(np.floor((DUR_MAX - DUR_MIN) / DURATION_STEP_NS)) + 1

# 16b-style amplitude sweep to find the π-phase amplitude at the 16a duration.
_16B_AMP_MIN = 0.05
_16B_AMP_MAX = 0.40
_16B_AMP_STEP = 0.002
_NOISE_16B = 0.02

# Fine amplitude window for error amplification (centered on 16b result).
AMPLITUDE_SPAN = 0.01
AMPLITUDE_STEP = 0.00025

RETURN_TO_INIT_NS = 16.0
_ERR_AMP_NOISE_STD = 0.01
_NOISE_16A = 0.01
# Padded cphase stack so ``n_gates`` is a JAX vmap batch axis (unrolled blocks).
# ``num_cphase_gates`` are multiples of 2 up to this value.
_MAX_CPHASE_BLOCKS = 60

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=250_000)}
QP_STUB = SimpleNamespace(name="q1_q2")

_SIG_X0 = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
_SIG_Y0 = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)

# Parity projector P_odd = |01⟩⟨01| + |10⟩⟨10|, rotated into the measurement
# basis after a closing X90 / Y90 on the target qubit (mode 0).
# P_odd_after_X90 = (Rx(π/2) ⊗ I)† P_odd (Rx(π/2) ⊗ I) — X-quadrature
# P_odd_after_Y90 = (Ry(π/2) ⊗ I)† P_odd (Ry(π/2) ⊗ I) — Y-quadrature
_I2 = jnp.eye(2, dtype=jnp.complex64)
_Rx_half_pi = jnp.array(
    [[1.0, -1j], [-1j, 1.0]], dtype=jnp.complex64
) / jnp.sqrt(2.0)
_Ry_half_pi = jnp.array(
    [[1.0, -1.0], [1.0, 1.0]], dtype=jnp.complex64
) / jnp.sqrt(2.0)
_P_ODD = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=jnp.complex64,
)
_U_x90 = jnp.kron(_Rx_half_pi, _I2)
_U_y90 = jnp.kron(_Ry_half_pi, _I2)
_P_ODD_AFTER_X90 = _U_x90.conj().T @ _P_ODD @ _U_x90
_P_ODD_AFTER_Y90 = _U_y90.conj().T @ _P_ODD @ _U_y90


def _simulate_sweep_bloch_eq_readout(
    device,
    make_schedule,
    tsave,
    *,
    seed: int = 42,
    noise_std: float = 0.0,
    **sweep_axes,
) -> np.ndarray:
    """Parity readout after closing X90/Y90 on target (two rows = I, Q).

    Computes P_odd = ⟨|01⟩⟨01| + |10⟩⟨10|⟩ of the two-qubit state after
    rotating by Rx(π/2) or Ry(π/2) on the target qubit (mode 0).  This gives
    0 for parallel spins (↑↑, ↓↓) and 1 for antiparallel spins (↑↓, ↓↑),
    matching the true parity readout of the qubit pair.
    """
    import dynamiqs as _dq

    psi0 = device.ground_state()
    obs_i = _dq.asqarray(_P_ODD_AFTER_X90)
    obs_q = _dq.asqarray(_P_ODD_AFTER_Y90)
    tsave_is_callable = callable(tsave)
    jump_ops = None

    def _inner(**kwargs):
        resolved = make_schedule(**kwargs)
        h_t = device.hamiltonian(resolved)
        ts = tsave(**kwargs) if tsave_is_callable else tsave
        sol = _vqpu_simulate(
            h_t,
            psi0,
            ts,
            solver="se",
            jump_ops=jump_ops,
            options=_SOLVER_KW,
        )
        e_i = _vqpu_expval(sol.states, obs_i)
        e_q = _vqpu_expval(sol.states, obs_q)
        return jnp.stack([e_i, e_q], axis=0)

    result = np.asarray(_vqpu_sweep(_inner, mode="outer", **sweep_axes))
    if noise_std > 0.0:
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(0, noise_std, size=result.shape)
        result = result + noise
        result = np.clip(result, 0.0, 1.0)
    return result


def _snap_duration_to_grid_4ns(t_pi_ns: float, grid: np.ndarray) -> float:
    """Pick the grid point (4 ns steps) closest to the fitted π-time."""
    if not np.isfinite(t_pi_ns):
        return float("nan")
    idx = int(np.argmin(np.abs(grid - t_pi_ns)))
    return float(grid[idx])


def run_16a_duration_sweep(
    device,
    *,
    exchange_amplitude_v: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
) -> dict[str, object]:
    """16a virtual sweep at fixed ``exchange_amplitude_v``: fit + raw traces for plots.

    Returns
    -------
    dict with keys: ``duration_grid_ns`` (1d), ``trace_ctrl0``, ``trace_ctrl1`` (1d,
    ``E_p2_given_p1_0`` vs duration), ``t_pi_fitted`` (ns), ``cz_duration_snapped`` (ns
    to nearest 4 ns grid), ``fit`` (duration-fit result dict for ``q1_q2``), and
    ``success`` (bool from fit).
    """
    pi_half_dur = float(DEFAULT_PULSE_DURATION_NS)
    pi_dur = float(DEFAULT_PULSE_DURATION_NS)

    duration_grid = DUR_MIN + DURATION_STEP_NS * np.arange(
        N_DURATIONS, dtype=np.float64
    )
    durations = jnp.asarray(duration_grid, dtype=jnp.float32)

    def tsave_phase0(exchange_duration, **_):
        dur = jnp.asarray(exchange_duration, dtype=jnp.float32)
        return jnp.stack(
            [
                jnp.zeros_like(dur, dtype=jnp.float32),
                2 * pi_half_dur + dur + RETURN_TO_INIT_NS,
            ],
            axis=-1,
        )

    def tsave_phase1(exchange_duration, **_):
        dur = jnp.asarray(exchange_duration, dtype=jnp.float32)
        return jnp.stack(
            [
                jnp.zeros_like(dur, dtype=jnp.float32),
                pi_dur + 2 * pi_half_dur + dur + RETURN_TO_INIT_NS,
            ],
            axis=-1,
        )

    def make_exp0(exchange_duration):
        sched = Schedule()
        ref0 = sched.play(
            GaussianIQPulse(
                duration=pi_half_dur,
                amplitude=pi_half_q0,
                frequency=q0_ghz,
                sigma=pi_half_dur / 5,
            ),
            channel="drive_q0",
        )
        ref_ex = sched.play(
            SquarePulse(
                duration=exchange_duration,
                amplitude=exchange_amplitude_v,
                frequency=0.0,
            ),
            channel="exchange_0_1",
            after=[ref0],
        )
        ref_idle = sched.play(
            SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
            channel="exchange_0_1",
            after=[ref_ex],
        )
        sched.play(
            GaussianIQPulse(
                duration=pi_half_dur,
                amplitude=pi_half_q0,
                frequency=q0_ghz,
                sigma=pi_half_dur / 5,
            ),
            channel="drive_q0",
            after=[ref_idle],
        )
        return sched.resolve()

    def make_exp1(exchange_duration):
        sched = Schedule()
        ref_pi = sched.play(
            GaussianIQPulse(
                duration=pi_dur,
                amplitude=pi_q1,
                frequency=q1_ghz,
                sigma=pi_dur / 5,
            ),
            channel="drive_q1",
        )
        ref_x90 = sched.play(
            GaussianIQPulse(
                duration=pi_half_dur,
                amplitude=pi_half_q0,
                frequency=q0_ghz,
                sigma=pi_half_dur / 5,
            ),
            channel="drive_q0",
            after=[ref_pi],
        )
        ref_ex = sched.play(
            SquarePulse(
                duration=exchange_duration,
                amplitude=exchange_amplitude_v,
                frequency=0.0,
            ),
            channel="exchange_0_1",
            after=[ref_x90],
        )
        ref_idle = sched.play(
            SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
            channel="exchange_0_1",
            after=[ref_ex],
        )
        sched.play(
            GaussianIQPulse(
                duration=pi_half_dur,
                amplitude=pi_half_q0,
                frequency=q0_ghz,
                sigma=pi_half_dur / 5,
            ),
            channel="drive_q0",
            after=[ref_idle],
        )
        return sched.resolve()

    r0 = simulate_sweep(
        device,
        make_exp0,
        tsave=tsave_phase0,
        observable_parity=True,
        noise_std=_NOISE_16A,
        seed=42,
        solver_options=_SOLVER_KW,
        exchange_duration=durations,
    )
    r1 = simulate_sweep(
        device,
        make_exp1,
        tsave=tsave_phase1,
        observable_parity=True,
        noise_std=_NOISE_16A,
        seed=43,
        solver_options=_SOLVER_KW,
        exchange_duration=durations,
    )

    pd0 = np.asarray(r0[..., -1], dtype=np.float64)
    pd1 = np.asarray(r1[..., -1], dtype=np.float64)
    data = np.stack([pd0, pd1], axis=0)

    ds_dur = build_joint_stream_analysis_ds(
        coords={
            "experiment_type": (
                np.array([0, 1], dtype=int),
                "experiment type",
                "",
            ),
            "exchange_duration": (
                np.asarray(durations, dtype=np.float64),
                "exchange duration",
                "ns",
            ),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )
    _, dur_fits = fit_duration_raw_data(
        ds_dur,
        [QP_STUB],
        exchange_amplitude=exchange_amplitude_v,
    )
    dur_fit = dur_fits["q1_q2"]
    t_pi = float(dur_fit["optimal_duration"])
    cz_dur = _snap_duration_to_grid_4ns(t_pi, duration_grid)
    return {
        "duration_grid_ns": np.asarray(duration_grid, dtype=np.float64),
        "trace_ctrl0": pd0,
        "trace_ctrl1": pd1,
        "t_pi_fitted": t_pi,
        "cz_duration_snapped": float(cz_dur) if np.isfinite(cz_dur) else float("nan"),
        "fit": dur_fit,
        "success": bool(dur_fit.get("success", False)),
    }


def _run_16a_virtual_sweep_for_pi_duration(
    device,
    *,
    exchange_amplitude_v: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
) -> tuple[float, np.ndarray]:
    """Return (snapped_cz_duration_ns, duration_grid) from a 16a-style virtual sweep + fit."""
    res = run_16a_duration_sweep(
        device,
        exchange_amplitude_v=exchange_amplitude_v,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
    )
    if not res["success"]:
        raise AssertionError(
            f"16a-style duration pre-fit failed: {res.get('fit', {})!r}"
        )
    t_pi = float(res["t_pi_fitted"])
    duration_grid = res["duration_grid_ns"]
    cz_dur = res["cz_duration_snapped"]
    assert np.isfinite(cz_dur), f"invalid snapped duration from t_pi={t_pi}"
    return float(cz_dur), np.asarray(duration_grid, dtype=np.float64)


# ── 16b-style amplitude sweep (single CZ pulse, variable amplitude) ──


def _make_16b_schedule_maker(
    control_state: int,
    cz_duration_ns: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
):
    """Schedule for 16b: state-prep + single exchange pulse + idle."""
    pi_half_dur = float(DEFAULT_PULSE_DURATION_NS)
    pi_dur = float(DEFAULT_PULSE_DURATION_NS)
    cz = float(cz_duration_ns)

    def make_sched(exchange_amplitude, **_k):
        sched = Schedule()
        if control_state == 0:
            ref_last = sched.play(
                GaussianIQPulse(
                    duration=pi_half_dur,
                    amplitude=pi_half_q0,
                    frequency=q0_ghz,
                    sigma=pi_half_dur / 5,
                ),
                channel="drive_q0",
            )
        else:
            r_pi = sched.play(
                GaussianIQPulse(
                    duration=pi_dur,
                    amplitude=pi_q1,
                    frequency=q1_ghz,
                    sigma=pi_dur / 5,
                ),
                channel="drive_q1",
            )
            ref_last = sched.play(
                GaussianIQPulse(
                    duration=pi_half_dur,
                    amplitude=pi_half_q0,
                    frequency=q0_ghz,
                    sigma=pi_half_dur / 5,
                ),
                channel="drive_q0",
                after=[r_pi],
            )
        ref_ex = sched.play(
            SquarePulse(
                duration=cz,
                amplitude=exchange_amplitude,
                frequency=0.0,
            ),
            channel="exchange_0_1",
            after=[ref_last],
        )
        sched.play(
            SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
            channel="exchange_0_1",
            after=[ref_ex],
        )
        return sched.resolve()

    return make_sched


def _make_16b_tsave(
    control_state: int,
    pi_half_dur: float,
    pi_dur: float,
    cz_duration_ns: float,
):
    """Fixed-total-time tsave for the 16b amplitude sweep."""
    if control_state == 0:
        total = pi_half_dur + float(cz_duration_ns) + RETURN_TO_INIT_NS
    else:
        total = pi_dur + pi_half_dur + float(cz_duration_ns) + RETURN_TO_INIT_NS

    def tsave(exchange_amplitude, **_k):
        ex = jnp.asarray(exchange_amplitude, dtype=jnp.float32)
        return jnp.stack(
            [
                jnp.zeros_like(ex, dtype=jnp.float32),
                jnp.full_like(ex, total, dtype=jnp.float32),
            ],
            axis=-1,
        )

    return tsave


def run_16b_amplitude_sweep(
    device,
    *,
    cz_duration_ns: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
    amp_min: float = _16B_AMP_MIN,
    amp_max: float = _16B_AMP_MAX,
    amp_step: float = _16B_AMP_STEP,
    noise_std: float = _NOISE_16B,
) -> dict[str, object]:
    """16b-style virtual amplitude sweep at fixed CZ duration.

    Returns
    -------
    dict with keys: ``amplitude_grid`` (1-d), ``optimal_amplitude`` (V),
    ``fit`` (amplitude-fit result dict for ``q1_q2``), and ``success``.
    """
    pi_half_dur = float(DEFAULT_PULSE_DURATION_NS)
    pi_dur = float(DEFAULT_PULSE_DURATION_NS)

    amplitude_grid = np.arange(
        amp_min, amp_max + 0.5 * amp_step, amp_step, dtype=np.float64,
    )
    amps_j = jnp.asarray(amplitude_grid, dtype=jnp.float32)

    results_by_ctrl: dict[int, np.ndarray] = {}
    for ctrl in (0, 1):
        r = _simulate_sweep_bloch_eq_readout(
            device,
            _make_16b_schedule_maker(
                ctrl, cz_duration_ns, pi_half_q0, pi_q1, q0_ghz, q1_ghz,
            ),
            tsave=_make_16b_tsave(ctrl, pi_half_dur, pi_dur, cz_duration_ns),
            seed=50 + ctrl,
            noise_std=noise_std,
            exchange_amplitude=amps_j,
        )
        r = np.asarray(r, dtype=np.float64)
        if r.ndim > 2 and r.shape[-1] > 1:
            r = r[..., -1]
        results_by_ctrl[ctrl] = r

    n_amp = len(amplitude_grid)
    data = np.zeros((2, 2, n_amp), dtype=np.float64)
    for ctrl in (0, 1):
        data[ctrl, 0, :] = results_by_ctrl[ctrl][:, 0]
        data[ctrl, 1, :] = results_by_ctrl[ctrl][:, 1]

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "control_state": (np.array([0, 1], dtype=int), "control state", ""),
            "analysis_axis": (
                np.array([0, 1], dtype=int),
                "analysis quadrature",
                "",
            ),
            "exchange_amplitude": (amplitude_grid, "barrier gate voltage", "V"),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )
    _, amp_fits = fit_amplitude_raw_data(
        ds_raw,
        [QP_STUB],
        exchange_duration=cz_duration_ns,
        quadrature_signal_center=0.5,
    )
    amp_fit = amp_fits["q1_q2"]
    opt_amp = float(amp_fit["optimal_amplitude"])
    return {
        "amplitude_grid": amplitude_grid,
        "optimal_amplitude": opt_amp,
        "fit": amp_fit,
        "success": bool(amp_fit.get("success", False)),
    }


def _make_padded_cphase_schedule_maker(
    control_state: int,
    cz_duration_ns: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
):
    """Return ``make(n_gates, exchange_amplitude)`` with a fixed, JAX-vmap’able
    cphase stack: ``_MAX_CPHASE_BLOCKS`` exchange+idle steps, each gated by
    ``(block_index < n_gates)`` so ``n_gates`` is a batched axis (``jax.vmap``).
    """
    pi_half_dur = float(DEFAULT_PULSE_DURATION_NS)
    pi_dur = float(DEFAULT_PULSE_DURATION_NS)
    cz = float(cz_duration_ns)

    def make_sched(n_gates, exchange_amplitude, **_k):
        sched = Schedule()
        if control_state == 0:
            ref_last = sched.play(
                GaussianIQPulse(
                    duration=pi_half_dur,
                    amplitude=pi_half_q0,
                    frequency=q0_ghz,
                    sigma=pi_half_dur / 5,
                ),
                channel="drive_q0",
            )
        else:
            r_pi = sched.play(
                GaussianIQPulse(
                    duration=pi_dur,
                    amplitude=pi_q1,
                    frequency=q1_ghz,
                    sigma=pi_dur / 5,
                ),
                channel="drive_q1",
            )
            ref_last = sched.play(
                GaussianIQPulse(
                    duration=pi_half_dur,
                    amplitude=pi_half_q0,
                    frequency=q0_ghz,
                    sigma=pi_half_dur / 5,
                ),
                channel="drive_q0",
                after=[r_pi],
            )
        n_g = jnp.asarray(n_gates, dtype=jnp.int32)
        for b in range(_MAX_CPHASE_BLOCKS):
            mask = jnp.int32(b) < n_g
            dur_ex = jnp.where(
                mask,
                jnp.asarray(cz, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            a_ex = jnp.where(
                mask, exchange_amplitude, jnp.asarray(0.0, dtype=jnp.float32)
            )
            r_ex = sched.play(
                SquarePulse(
                    duration=dur_ex,
                    amplitude=a_ex,
                    frequency=0.0,
                ),
                channel="exchange_0_1",
                after=[ref_last],
            )
            idur = jnp.where(
                mask,
                jnp.asarray(RETURN_TO_INIT_NS, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            ref_last = sched.play(
                SquarePulse(duration=idur, amplitude=0.0, frequency=0.0),
                channel="exchange_0_1",
                after=[r_ex],
            )
        return sched.resolve()

    return make_sched


def _make_tsave_padded_pre_readout(
    control_state: int,
    pi_half_dur: float,
    pi_dur: float,
    cz_duration_ns: float,
):
    block = float(cz_duration_ns) + RETURN_TO_INIT_NS

    def tsave(n_gates, exchange_amplitude, **kw):
        del kw
        ex = jnp.asarray(exchange_amplitude, dtype=jnp.float32)
        n = jnp.asarray(n_gates, dtype=jnp.float32)
        if control_state == 0:
            base = float(pi_half_dur)
        else:
            base = float(pi_dur) + float(pi_half_dur)
        tend = base + n * block
        return jnp.stack(
            [
                jnp.zeros_like(ex, dtype=jnp.float32),
                jnp.full_like(ex, tend, dtype=jnp.float32),
            ],
            axis=-1,
        )

    return tsave


def _sweep_cphase_bloch_for_control(
    device,
    *,
    control_state: int,
    cz_duration_ns: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
    pi_half_dur: float,
    pi_dur: float,
    n_gates_arr: Any,
    amplitudes_arr: Any,
    seed: int,
    noise_std: float,
) -> np.ndarray:
    """(n_gate_points, n_amp, 2) Bloch I/Q in [0,1] at pre-readout time."""
    r = _simulate_sweep_bloch_eq_readout(
        device,
        _make_padded_cphase_schedule_maker(
            control_state,
            cz_duration_ns,
            pi_half_q0,
            pi_q1,
            q0_ghz,
            q1_ghz,
        ),
        tsave=_make_tsave_padded_pre_readout(
            control_state, pi_half_dur, pi_dur, cz_duration_ns
        ),
        seed=seed,
        noise_std=noise_std,
        n_gates=n_gates_arr,
        exchange_amplitude=amplitudes_arr,
    )
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-1] > 1:
        r = r[..., -1]
    return r


@pytest.mark.analysis
def test_17_geometric_cz_error_amplification_analysis(
    ld_device,
    calibrated_pi_half_amp,
    calibrated_control_pi_amp,
    analysis_runner,
    save_analysis_plot,
):
    """virtual_qpu error-amplification: 16a duration -> 16b amplitude -> 17 fit."""
    device = ld_device
    q0_ghz = float(device.params.qubit_freqs[0])
    q1_ghz = float(device.params.qubit_freqs[1])
    pi_half_q0 = float(calibrated_pi_half_amp)
    pi_q1 = float(calibrated_control_pi_amp)
    pi_half_dur = float(DEFAULT_PULSE_DURATION_NS)
    pi_dur = float(DEFAULT_PULSE_DURATION_NS)

    # ── Step 1: 16a duration sweep → snapped CZ duration ──────────────
    cz_duration_ns, _ = _run_16a_virtual_sweep_for_pi_duration(
        device,
        exchange_amplitude_v=EXCHANGE_AMPLITUDE_REF_V,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
    )
    assert cz_duration_ns % 4.0 == pytest.approx(
        0.0
    ), "CZ duration must be a multiple of 4 ns"

    # ── Step 2: 16b amplitude sweep at that duration → amplitude center ──
    res_16b = run_16b_amplitude_sweep(
        device,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
    )
    assert res_16b["success"], f"16b amplitude pre-fit failed: {res_16b['fit']}"
    amplitude_center = float(res_16b["optimal_amplitude"])
    assert np.isfinite(amplitude_center), "16b returned invalid amplitude"
    assert _16B_AMP_MIN <= amplitude_center <= _16B_AMP_MAX, (
        f"16b amplitude {amplitude_center:.4f} V outside sweep range"
    )

    # ── Step 3: error-amplification sweep centred on 16b result ────────
    amplitudes = np.arange(
        amplitude_center - AMPLITUDE_SPAN,
        amplitude_center + AMPLITUDE_SPAN + 0.5 * AMPLITUDE_STEP,
        AMPLITUDE_STEP,
        dtype=np.float64,
    )
    num_gates = np.arange(2, _MAX_CPHASE_BLOCKS + 1, 2, dtype=np.int32)
    n_amp = len(amplitudes)
    n_n = len(num_gates)

    amps_j = jnp.asarray(amplitudes, dtype=jnp.float32)
    ng_j = jnp.asarray(num_gates, dtype=jnp.int32)

    data = np.zeros((2, 2, n_amp, n_n), dtype=np.float64)
    base_seed = 200
    r0 = _sweep_cphase_bloch_for_control(
        device,
        control_state=0,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
        pi_half_dur=pi_half_dur,
        pi_dur=pi_dur,
        n_gates_arr=ng_j,
        amplitudes_arr=amps_j,
        seed=base_seed,
        noise_std=_ERR_AMP_NOISE_STD,
    )
    r1 = _sweep_cphase_bloch_for_control(
        device,
        control_state=1,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
        pi_half_dur=pi_half_dur,
        pi_dur=pi_dur,
        n_gates_arr=ng_j,
        amplitudes_arr=amps_j,
        seed=base_seed + 1,
        noise_std=_ERR_AMP_NOISE_STD,
    )
    data[0, 0, :, :] = r0[:, :, 0].T
    data[0, 1, :, :] = r0[:, :, 1].T
    data[1, 0, :, :] = r1[:, :, 0].T
    data[1, 1, :, :] = r1[:, :, 1].T

    assert data.shape == (2, 2, n_amp, len(num_gates))

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "control_state": (
                np.array([0, 1], dtype=int),
                "control state",
                "",
            ),
            "analysis_axis": (
                np.array([0, 1], dtype=int),
                "analysis quadrature",
                "",
            ),
            "exchange_amplitude": (
                amplitudes,
                "barrier gate voltage",
                "V",
            ),
            "num_cphase_gates": (
                num_gates,
                "number of raw CPhase gates",
                "",
            ),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "exchange_amplitude_center": amplitude_center,
            "exchange_amplitude_span": AMPLITUDE_SPAN,
            "exchange_amplitude_step": AMPLITUDE_STEP,
            "max_num_cphase_gates": int(num_gates.max()),
            "quadrature_signal_center": 0.5,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"Fit should succeed: {fit}"
    assert DUR_MIN <= cz_duration_ns <= DUR_MAX
    assert np.isfinite(fit["optimal_amplitude"])
    assert (
        abs(fit["optimal_amplitude"] - amplitude_center)
        <= AMPLITUDE_SPAN + 2 * AMPLITUDE_STEP
    )

    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "chevron_signal_q1_q2" in ds_fit.data_vars
    assert ds_fit["chevron_signal_q1_q2"].dims == (
        "exchange_amplitude",
        "num_cphase_gates",
    )
    assert "mean_signal_q1_q2" in ds_fit.data_vars
    assert "mean_signal_fit_q1_q2" in ds_fit.data_vars
    assert "figure" in node.results

    assert "fig_raw_panels" in node.results
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    save_analysis_plot(node.results["fig_raw_panels"], artifacts_dir, "raw_state_panels.png")
    assert (artifacts_dir / "raw_state_panels.png").exists()
