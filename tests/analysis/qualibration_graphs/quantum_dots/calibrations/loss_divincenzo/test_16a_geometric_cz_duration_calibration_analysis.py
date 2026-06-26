"""Analysis test for 16a_geometric_cz_duration_calibration.

Uses virtual_qpu to simulate the exchange dynamics from the
fixed-amplitude CZ duration node.  State preparation (X90 on target,
π on control) and readout (closing X90) use perfect unitary gates
via matrix multiplication; only the exchange interaction is
time-evolved through the ODE solver.

The I quadrature comes from the rotated observable (closing X90
folded into the parity projector); the Q quadrature is synthesised
via Hilbert transform so that the dataset matches the real hardware
format (control_state × analysis_axis × exchange_duration).
"""

from __future__ import annotations

import numpy as np
import pytest

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from virtual_qpu import prepare_bloch_trajectories, plot_bloch_trajectories
from virtual_qpu.dynamics import simulate as vqpu_simulate
from virtual_qpu.operators import expval as vqpu_expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as vqpu_sweep

from .conftest import (
    ARTIFACTS_BASE,
    QUBIT_PAIR_NAMES,
    build_joint_stream_analysis_ds,
)

NODE_NAME = "16a_geometric_cz_duration_calibration"

EXCHANGE_AMPLITUDE = 0.23
DURATION_STEP_NS = 20.0
DUR_MIN = 16.0
DUR_MAX = 20000.0
N_DURATIONS = int(np.floor((DUR_MAX - DUR_MIN) / DURATION_STEP_NS)) + 1
NOISE_STD = 0.1
RETURN_TO_INIT_NS = 16.0
SIGNAL_CENTER = 0.5
BLOCH_N_TIMEPOINTS = 2000

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=1_000_000)}

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

# X90 on target qubit (mode 0): Rx(π/2) ⊗ I₂
_U_X90_Q0 = jnp.kron(_Rx_half_pi, _I2)

# π on control qubit (mode 1): I₂ ⊗ Rx(π)
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

# Closing X90 on q0 folded into observable: U_x90† P_odd U_x90
_OBS_PARITY_AFTER_X90 = dq.asqarray(
    _U_X90_Q0.conj().T @ _P_ODD @ _U_X90_Q0
)


def _synthesise_quadrature(i_signal: np.ndarray, center: float = 0.5) -> np.ndarray:
    """Synthesise the Y90 (Q) quadrature from the X90 (I) signal via Hilbert transform."""
    analytic = hilbert(i_signal - center)
    return center + analytic.imag


def _make_exchange_schedule(exchange_duration):
    """Exchange-only schedule: exchange pulse + return-to-init idle."""
    sched = Schedule()
    ref_ex = sched.play(
        SquarePulse(
            duration=exchange_duration,
            amplitude=EXCHANGE_AMPLITUDE,
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
def test_16a_geometric_cz_duration_analysis(
    ld_device,
    analysis_runner,
):
    """Fixed-amplitude cphase virtual_qpu sweep -> extract pi phase time.

    State preparation and readout use perfect unitary gates; only the
    exchange interaction is time-evolved through the ODE solver.
    """
    device = ld_device

    duration_grid = DUR_MIN + DURATION_STEP_NS * np.arange(
        N_DURATIONS, dtype=np.float64
    )
    durations = jnp.asarray(duration_grid, dtype=jnp.float32)

    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    def _sweep_exchange(psi_init, seed):
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

        result = np.asarray(vqpu_sweep(_inner, exchange_duration=durations))
        rng = np.random.default_rng(seed=seed)
        result = result + rng.normal(0, NOISE_STD, size=result.shape)
        return np.clip(result, 0.0, 1.0)

    r0 = _sweep_exchange(psi_ctrl0, seed=42)
    r1 = _sweep_exchange(psi_ctrl1, seed=43)

    i0 = np.asarray(r0[..., -1], dtype=np.float64)
    i1 = np.asarray(r1[..., -1], dtype=np.float64)
    q0 = _synthesise_quadrature(i0, center=SIGNAL_CENTER)
    q1 = _synthesise_quadrature(i1, center=SIGNAL_CENTER)

    # shape: (control_state=2, analysis_axis=2, exchange_duration=N)
    data = np.stack([
        np.stack([i0, q0], axis=0),
        np.stack([i1, q1], axis=0),
    ], axis=0)
    assert data.shape == (2, 2, N_DURATIONS)

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
            "exchange_duration": (
                np.asarray(durations, dtype=np.float64),
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
            "min_exchange_duration_in_ns": int(DUR_MIN),
            "max_exchange_duration_in_ns": int(np.ceil(duration_grid[-1])) + 1,
            "duration_step_in_ns": int(DURATION_STEP_NS),
            "quadrature_signal_center": SIGNAL_CENTER,
        },
        namespace_overrides={
            "exchange_amplitude": EXCHANGE_AMPLITUDE,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"Analysis should succeed: {fit}"
    assert fit["exchange_amplitude"] == pytest.approx(EXCHANGE_AMPLITUDE)
    assert DUR_MIN <= fit["optimal_duration"] <= duration_grid[-1] + DURATION_STEP_NS

    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "conditional_phase_q1_q2" in ds_fit.data_vars
    assert "phi_ctrl_ground_q1_q2" in ds_fit.data_vars
    assert "phi_ctrl_excited_q1_q2" in ds_fit.data_vars
    assert "t_pi_cphase_q1_q2" in ds_fit.data_vars

    assert "figure" in node.results

    # -- Bloch sphere at the fitted optimal exchange duration -------------
    optimal_dur = fit["optimal_duration"]
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    resolved_ex = _make_exchange_schedule(optimal_dur)
    H_ex = device.hamiltonian(resolved_ex)
    ex_total = float(resolved_ex.total_duration)

    ts = jnp.linspace(0.0, ex_total, BLOCH_N_TIMEPOINTS)
    res0 = vqpu_simulate(H_ex, psi_ctrl0, ts, solver="se", options=_SOLVER_KW)
    res1 = vqpu_simulate(H_ex, psi_ctrl1, ts, solver="se", options=_SOLVER_KW)

    bloch_labels = ["target (q0)", "control (q1)"]
    b0 = prepare_bloch_trajectories(
        ts, res0.states, device=device, labels=bloch_labels,
    )
    b1 = prepare_bloch_trajectories(
        ts, res1.states, device=device, labels=bloch_labels,
    )
    fig_bloch, _ = plot_bloch_trajectories(
        [b0, b1],
        backend="matplotlib",
        title=(
            f"CPhase exchange at optimal duration {optimal_dur:.0f} ns "
            f"(\u03c0 cphase)"
        ),
        series_labels=["ctrl |0\u27E9", "ctrl |1\u27E9"],
    )
    bloch_path = artifacts_dir / "bloch_optimal_duration.png"
    fig_bloch.savefig(bloch_path, dpi=150)
    plt.close(fig_bloch)
    assert bloch_path.exists()


# ---------------------------------------------------------------------------
# Bloch-sphere trajectories at representative exchange durations
# ---------------------------------------------------------------------------

BLOCH_EXCHANGE_DURATIONS = [100.0, 400.0, 800.0]


@pytest.mark.analysis
def test_16a_bloch_sphere_trajectories(
    ld_device,
):
    """Bloch-sphere trajectories for the cphase exchange: ctrl |0> vs ctrl |1>.

    For each representative exchange duration, only the exchange interaction
    is time-evolved; state preparation (X90, π) uses perfect unitary gates.
    The Bloch trajectories show the conditional-phase accumulation on the
    target qubit and the control-qubit state.
    """
    device = ld_device
    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for ex_dur in BLOCH_EXCHANGE_DURATIONS:
        resolved_ex = _make_exchange_schedule(ex_dur)
        H_ex = device.hamiltonian(resolved_ex)
        ex_total = float(resolved_ex.total_duration)

        tsave = jnp.linspace(0.0, ex_total, BLOCH_N_TIMEPOINTS)

        result0 = vqpu_simulate(
            H_ex, psi_ctrl0, tsave, solver="se", options=_SOLVER_KW,
        )
        result1 = vqpu_simulate(
            H_ex, psi_ctrl1, tsave, solver="se", options=_SOLVER_KW,
        )

        bloch_labels = ["target (q0)", "control (q1)"]
        bloch0 = prepare_bloch_trajectories(
            tsave, result0.states, device=device, labels=bloch_labels,
        )
        bloch1 = prepare_bloch_trajectories(
            tsave, result1.states, device=device, labels=bloch_labels,
        )

        fig, _ = plot_bloch_trajectories(
            [bloch0, bloch1],
            backend="matplotlib",
            title=(
                f"CPhase exchange Bloch trajectories "
                f"\u2014 exchange {ex_dur:.0f} ns"
            ),
            series_labels=[
                "ctrl |0\u27E9",
                "ctrl |1\u27E9",
            ],
        )
        out_path = artifacts_dir / f"bloch_exchange_{ex_dur:.0f}ns.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        assert out_path.exists(), f"Bloch plot not created: {out_path}"
