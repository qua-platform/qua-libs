"""Analysis test for 16_geometric_cz_calibration — quadrature readout.

Uses virtual_qpu (LossDiVincenzoDevice + dynamiqs) to simulate the four
quadrature experiments (2 control states × 2 analysis axes) sweeping
exchange amplitude × duration.  State preparation (X90 on target, π on
control) and readout (closing X90) use perfect unitary gates; only the
exchange interaction is time-evolved through the ODE solver.

The I quadrature comes from the rotated parity observable (closing X90
folded into the projector); the Q quadrature is synthesised via Hilbert
transform along the duration axis for each amplitude slice.

The analysis extracts the conditional phase phi1-phi0-pi vs (V, t) and finds
the amplitude V* where it crosses pi at the user-specified target duration.
"""

from __future__ import annotations

import numpy as np
import pytest

import dynamiqs as dq
import jax.numpy as jnp
from scipy.signal import hilbert

from virtual_qpu.dynamics import simulate as vqpu_simulate
from virtual_qpu.operators import expval as vqpu_expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as vqpu_sweep

from .conftest import (
    QUBIT_PAIR_NAMES,
    build_joint_stream_analysis_ds,
)

NODE_NAME = "16_geometric_cz_calibration"

# 2D sweep grid (kept small for test speed)
N_VOLTAGES = 25
V_MIN = 0.15
V_MAX = 0.33

DURATION_STEP_NS = 8.0
DUR_MIN = 16.0
DUR_MAX = 1000.0
_DUR_STEPS = int(np.floor((DUR_MAX - DUR_MIN) / DURATION_STEP_NS))
DUR_LAST_SWEPT = DUR_MIN + _DUR_STEPS * DURATION_STEP_NS
N_DURATIONS = _DUR_STEPS + 1

NOISE_STD = 0.01
SIGNAL_CENTER = 0.5
TARGET_DURATION_NS = 100

RETURN_TO_INIT_NS = 16.0
_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=250_000)}

# ---------------------------------------------------------------------------
# Perfect unitary gates (4×4 two-qubit Hilbert space)
# ---------------------------------------------------------------------------
_I2 = jnp.eye(2, dtype=jnp.complex64)

_Rx_half_pi = jnp.array(
    [[1.0, -1j], [-1j, 1.0]], dtype=jnp.complex64
) / jnp.sqrt(2.0)

_Rx_pi = jnp.array(
    [[0.0, -1j], [-1j, 0.0]], dtype=jnp.complex64
)

_U_X90_Q0 = jnp.kron(_Rx_half_pi, _I2)
_U_PI_Q1 = jnp.kron(_I2, _Rx_pi)

_P_ODD = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=jnp.complex64,
)

_OBS_PARITY_AFTER_X90 = dq.asqarray(
    _U_X90_Q0.conj().T @ _P_ODD @ _U_X90_Q0
)


def _synthesise_quadrature_2d(
    i_signal_2d: np.ndarray,
    center: float = 0.5,
) -> np.ndarray:
    """Hilbert-synthesise Q from I along the duration axis (axis=-1) per amplitude slice."""
    q = np.empty_like(i_signal_2d)
    for i in range(i_signal_2d.shape[0]):
        analytic = hilbert(i_signal_2d[i] - center)
        q[i] = center + analytic.imag
    return q


def _make_exchange_schedule(exchange_voltage, exchange_duration):
    sched = Schedule()
    ref_ex = sched.play(
        SquarePulse(
            duration=exchange_duration,
            amplitude=exchange_voltage,
            frequency=0.0,
        ),
        channel="exchange_0_1",
    )
    sched.play(
        SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
        channel="exchange_0_1",
        after=[ref_ex],
    )
    return sched.resolve()


@pytest.mark.analysis
def test_16_geometric_cz_analysis(
    ld_device,
    analysis_runner,
):
    """Geometric CZ quadrature: 2D virtual_qpu sweep → extract (V*, t*) via analysis."""
    device = ld_device

    voltages_np = np.linspace(V_MIN, V_MAX, N_VOLTAGES, dtype=np.float64)
    voltages = jnp.asarray(voltages_np, dtype=jnp.float32)

    duration_grid = DUR_MIN + DURATION_STEP_NS * np.arange(
        N_DURATIONS, dtype=np.float64
    )
    assert np.isclose(duration_grid[-1], DUR_LAST_SWEPT)
    durations = jnp.asarray(duration_grid, dtype=jnp.float32)

    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    def _sweep_exchange_2d(psi_init, seed):
        def _inner(**kwargs):
            resolved = _make_exchange_schedule(**kwargs)
            H_t = device.hamiltonian(resolved)
            dur = jnp.asarray(kwargs["exchange_duration"], dtype=jnp.float32)
            ts = jnp.stack([
                jnp.float32(0.0),
                dur + jnp.float32(RETURN_TO_INIT_NS),
            ])
            sol = vqpu_simulate(
                H_t, psi_init, ts, solver="se", options=_SOLVER_KW,
            )
            return vqpu_expval(sol.states, _OBS_PARITY_AFTER_X90)

        result = np.asarray(vqpu_sweep(
            _inner,
            exchange_voltage=voltages,
            exchange_duration=durations,
        ))
        rng = np.random.default_rng(seed=seed)
        result = result + rng.normal(0, NOISE_STD, size=result.shape)
        return np.clip(result, 0.0, 1.0)

    r0 = _sweep_exchange_2d(psi_ctrl0, seed=42)
    r1 = _sweep_exchange_2d(psi_ctrl1, seed=43)

    # r0, r1 shape: (n_amps, n_durs, 2_tsave); take final time point
    i0 = np.asarray(r0[..., -1], dtype=np.float64)
    i1 = np.asarray(r1[..., -1], dtype=np.float64)
    assert i0.shape == (N_VOLTAGES, N_DURATIONS)

    q0 = _synthesise_quadrature_2d(i0, center=SIGNAL_CENTER)
    q1 = _synthesise_quadrature_2d(i1, center=SIGNAL_CENTER)

    # shape: (control_state=2, analysis_axis=2, n_amps, n_durs)
    data = np.stack([
        np.stack([i0, q0], axis=0),
        np.stack([i1, q1], axis=0),
    ], axis=0)
    assert data.shape == (2, 2, N_VOLTAGES, N_DURATIONS)

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
                voltages_np,
                "barrier gate voltage",
                "V",
            ),
            "exchange_duration": (
                np.asarray(duration_grid, dtype=np.float64),
                "exchange duration",
                "ns",
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
            "target_duration_ns": TARGET_DURATION_NS,
            "quadrature_signal_center": SIGNAL_CENTER,
            "min_exchange_amplitude": float(V_MIN),
            "max_exchange_amplitude": float(V_MAX),
            "amplitude_step": float((V_MAX - V_MIN) / max(N_VOLTAGES - 1, 1)),
            "min_exchange_duration_in_ns": int(DUR_MIN),
            "max_exchange_duration_in_ns": int(np.ceil(DUR_LAST_SWEPT)) + 1,
            "duration_step_in_ns": int(DURATION_STEP_NS),
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "fit_results" in node.results, "analyse_data should populate fit_results"

    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"Analysis should succeed: {fit}"
    assert np.isfinite(fit["optimal_amplitude"])
    assert np.isfinite(fit["optimal_duration"])
    assert V_MIN <= fit["optimal_amplitude"] <= V_MAX
    assert fit["optimal_duration"] == pytest.approx(TARGET_DURATION_NS, abs=DURATION_STEP_NS)

    assert "ds_fit" in node.results, "analyse_data should produce ds_fit"
    ds_fit = node.results["ds_fit"]
    assert "conditional_phase_q1_q2" in ds_fit.data_vars
    assert "phi_ctrl_ground_q1_q2" in ds_fit.data_vars
    assert "phi_ctrl_excited_q1_q2" in ds_fit.data_vars
    assert "t_pi_cphase_q1_q2" in ds_fit.data_vars
    assert "cond_phase_at_target_q1_q2" in ds_fit.data_vars

    assert "figure" in node.results, "plot_data should produce a figure"
