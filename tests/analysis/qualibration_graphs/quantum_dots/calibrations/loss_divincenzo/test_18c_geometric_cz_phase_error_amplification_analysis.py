"""Analysis test for 18c_geometric_cz_phase_error_amplification.

Uses virtual_qpu (Loss-DiVincenzo Hamiltonian + dynamiqs) for the exchange
interaction and perfect unitary gates (matrix multiplication) for state
preparation and the analysis rotation.

Simulation strategy
-------------------
Two independent Ramsey sweeps are simulated at V* (found by a 16b-style
pre-flight):

  Sweep A — target qubit phase sweep
    Initial states: X90(target)|00⟩  and  X90(target)·Xπ(control)|00⟩
    Analysis π/2: θ-rotation on target qubit
    Measured observable: P_odd folded through U_target(θ)

  Sweep B — control qubit phase sweep
    Initial states: X90(control)|00⟩  and  X90(control)·Xπ(target)|00⟩
    Analysis π/2: θ-rotation on control qubit
    Measured observable: P_odd folded through U_control(θ)

Each sweep gives a (2, N_gates, N_phases) tensor which is added to ds_raw
as E_p2_given_p1_0_{name} (target) and E_p2_given_p1_0_ctrl_{name} (control).

Test flow
---------
1. 16b-style pre-flight: find the true CZ amplitude V*.
2. Simulate both sweeps for each N ∈ {2, 4, …, MAX_NUM_CPHASE_GATES}.
3. Assemble ds_raw with both sweep variables.
4. Run the 18c analysis via analysis_runner.
5. Assert fit_results contains target and control phase compensations.
6. Physics check: ⟨|D(N,θ)|⟩_θ is flat at V* for both sweeps.
7. Verify update_state wrote phase_shift_target / phase_shift_control to the CZ macro.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import dynamiqs as dq
import jax.numpy as jnp

from virtual_qpu.dynamics import simulate as vqpu_simulate
from virtual_qpu.operators import expval as vqpu_expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as vqpu_sweep

from calibration_utils.geometric_cz_amplitude.analysis import (
    fit_raw_data as fit_16b_amplitude,
)

from .conftest import (
    ARTIFACTS_BASE,
    DEFAULT_PULSE_DURATION_NS,
    QUBIT_PAIR_NAMES,
    build_joint_stream_analysis_ds,
    simulate_sweep,
)

# ── Node & sweep configuration ─────────────────────────────────────────────

NODE_NAME = "18c_geometric_cz_phase_error_amplification"
QP_STUB = SimpleNamespace(name="q1_q2")

FIXED_DURATION_NS = 50.0
RETURN_TO_INIT_NS = 16.0

AMP_MIN = 0.05
AMP_MAX = 0.40
AMP_STEP = 0.005

MAX_NUM_CPHASE_GATES = 20  # sweep: 2, 4, 6, 8, 10
N_PHASES = 51
NOISE_STD = 0.02
SIGNAL_CENTER = 0.5

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=1_000_000)}

# ── Perfect single-qubit unitaries in the 4-qubit Hilbert space ──────────────
# Basis order: |Q0, Q1⟩ = |target, control⟩  (kron product = target ⊗ control)

_I2 = jnp.eye(2, dtype=jnp.complex64)

_Rx_half_pi = jnp.array(
    [[1.0, -1j], [-1j, 1.0]], dtype=jnp.complex64
) / jnp.sqrt(2.0)

_Rx_pi = jnp.array(
    [[0.0, -1j], [-1j, 0.0]], dtype=jnp.complex64
)

# X90 on target (mode 0): Rx(π/2) ⊗ I₂
_U_X90_Q0 = jnp.kron(_Rx_half_pi, _I2)
# X90 on control (mode 1): I₂ ⊗ Rx(π/2)
_U_X90_Q1 = jnp.kron(_I2, _Rx_half_pi)

# π on target (mode 0): Rx(π) ⊗ I₂
_U_PI_Q0 = jnp.kron(_Rx_pi, _I2)
# π on control (mode 1): I₂ ⊗ Rx(π)
_U_PI_Q1 = jnp.kron(_I2, _Rx_pi)

# Two-qubit parity projector P_odd = |01⟩⟨01| + |10⟩⟨10|
_P_ODD = jnp.array(
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    dtype=jnp.complex64,
)

# ── Phase-dependent folded observables ────────────────────────────────────────

_PHASE_FRACS = np.linspace(0, 1, N_PHASES, endpoint=False)
_PHASE_RADS = _PHASE_FRACS * 2 * np.pi


def _phase_rotation_matrix(theta_rad: float) -> jnp.ndarray:
    """π/2 analysis rotation: U(θ) = (1/√2)[[1,-ie^{-iθ}],[-ie^{iθ},1]]."""
    return jnp.array(
        [
            [1.0, -1j * jnp.exp(-1j * theta_rad)],
            [-1j * jnp.exp(1j * theta_rad), 1.0],
        ],
        dtype=jnp.complex64,
    ) / jnp.sqrt(2.0)


def _make_phase_obs_target(theta_rad: float) -> jnp.ndarray:
    """P_obs(θ) = (U_target(θ) ⊗ I₂)† P_odd (U_target(θ) ⊗ I₂)."""
    U_2q = jnp.kron(_phase_rotation_matrix(theta_rad), _I2)
    return U_2q.conj().T @ _P_ODD @ U_2q


def _make_phase_obs_control(theta_rad: float) -> jnp.ndarray:
    """P_obs(θ) = (I₂ ⊗ U_control(θ))† P_odd (I₂ ⊗ U_control(θ))."""
    U_2q = jnp.kron(_I2, _phase_rotation_matrix(theta_rad))
    return U_2q.conj().T @ _P_ODD @ U_2q


# Pre-computed (N_PHASES, 4, 4) observable tensors for batch einsum evaluation.
_OBS_TARGET = jnp.stack(
    [_make_phase_obs_target(float(th * 2 * np.pi)) for th in _PHASE_FRACS],
    axis=0,
)
_OBS_CONTROL = jnp.stack(
    [_make_phase_obs_control(float(th * 2 * np.pi)) for th in _PHASE_FRACS],
    axis=0,
)

# ── Schedule helpers ──────────────────────────────────────────────────────────


def _make_n_gate_schedule(n_gates: int, exchange_amplitude: float) -> object:
    """Exchange schedule with n_gates CZ blocks at fixed amplitude."""
    sched = Schedule()
    ref = None
    for _ in range(n_gates):
        ref = sched.play(
            SquarePulse(
                duration=FIXED_DURATION_NS,
                amplitude=exchange_amplitude,
                frequency=0.0,
            ),
            channel="exchange_0_1",
            after=[ref] if ref is not None else [],
        )
        ref = sched.play(
            SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
            channel="exchange_0_1",
            after=[ref],
        )
    return sched.resolve()


# ── Core simulation helpers ───────────────────────────────────────────────────


def _final_state(device, psi_init: jnp.ndarray, n_gates: int, v_star: float) -> jnp.ndarray:
    """Simulate n_gates CZ pulses at v_star; return squeezed final ket (4,)."""
    sched = _make_n_gate_schedule(n_gates, v_star)
    total_ns = n_gates * (FIXED_DURATION_NS + RETURN_TO_INIT_NS)
    tsave = jnp.array([0.0, total_ns], dtype=jnp.float32)
    H_t = device.hamiltonian(sched)
    sol = vqpu_simulate(H_t, psi_init, tsave, solver="se", options=_SOLVER_KW)
    raw = sol.states[-1]
    return jnp.squeeze(raw.to_jax() if hasattr(raw, "to_jax") else jnp.asarray(raw))


def _phase_signals_from_state(psi: jnp.ndarray, obs: jnp.ndarray) -> np.ndarray:
    """Expectation values of all phase observables for state psi.

    Args:
        psi: ket of shape (4,).
        obs: pre-computed observable tensor of shape (N_PHASES, 4, 4).

    Returns:
        Array of shape (N_PHASES,).
    """
    obs_psi = jnp.einsum("kij,j->ki", obs, psi)  # (N_PHASES, 4)
    return np.asarray(
        jnp.real(jnp.einsum("i,ki->k", jnp.conj(psi), obs_psi))
    )


def _sweep_n_gates_phase(
    device,
    psi_cond0: jnp.ndarray,
    psi_cond1: jnp.ndarray,
    v_star: float,
    num_cphase_gates: np.ndarray,
    obs: jnp.ndarray,
    *,
    noise_std: float = NOISE_STD,
    seed: int = 42,
) -> np.ndarray:
    """Simulate (2, N_gates, N_phases) for both conditional states.

    One ODE solve per N; the phase dimension is evaluated analytically from
    the final state via the pre-computed folded observables.
    """
    rows_cond0, rows_cond1 = [], []
    for n in num_cphase_gates:
        psi0f = _final_state(device, psi_cond0, int(n), v_star)
        psi1f = _final_state(device, psi_cond1, int(n), v_star)
        rows_cond0.append(_phase_signals_from_state(psi0f, obs))
        rows_cond1.append(_phase_signals_from_state(psi1f, obs))

    sig_cond0 = np.stack(rows_cond0, axis=0)  # (N_gate_counts, N_PHASES)
    sig_cond1 = np.stack(rows_cond1, axis=0)

    if noise_std > 0:
        rng = np.random.default_rng(seed=seed)
        for arr in [sig_cond0, sig_cond1]:
            arr += rng.normal(0, noise_std, size=arr.shape)
            np.clip(arr, 0.0, 1.0, out=arr)

    return np.stack([sig_cond0, sig_cond1], axis=0)  # (2, N_gate_counts, N_PHASES)


# ── 16b pre-flight: locate true CZ amplitude ─────────────────────────────────


def _run_16b_preflight(device, amplitudes: np.ndarray) -> float:
    """16b-style amplitude sweep (Hilbert-transform Q trick) → V*."""
    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    _obs_x90 = dq.asqarray(_U_X90_Q0.conj().T @ _P_ODD @ _U_X90_Q0)
    amplitudes_j = jnp.asarray(amplitudes, dtype=jnp.float32)
    tsave = jnp.array(
        [0.0, FIXED_DURATION_NS + RETURN_TO_INIT_NS], dtype=jnp.float32
    )

    def _sweep_ctrl(psi_init, seed_offset):
        def _inner(exchange_amplitude):
            sched = Schedule()
            ref = sched.play(
                SquarePulse(
                    duration=FIXED_DURATION_NS,
                    amplitude=exchange_amplitude,
                    frequency=0.0,
                ),
                channel="exchange_0_1",
            )
            sched.play(
                SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
                channel="exchange_0_1",
                after=[ref],
            )
            H_t = device.hamiltonian(sched.resolve())
            sol = vqpu_simulate(H_t, psi_init, tsave, solver="se", options=_SOLVER_KW)
            return vqpu_expval(sol.states, _obs_x90)

        r = np.asarray(vqpu_sweep(_inner, mode="outer", exchange_amplitude=amplitudes_j))
        rng = np.random.default_rng(seed=seed_offset)
        return np.clip(r + rng.normal(0, 0.03, size=r.shape), 0.0, 1.0)

    r0 = _sweep_ctrl(psi_ctrl0, seed_offset=10)
    r1 = _sweep_ctrl(psi_ctrl1, seed_offset=11)

    i0 = np.asarray(r0[..., -1], dtype=np.float64)
    i1 = np.asarray(r1[..., -1], dtype=np.float64)

    from scipy.signal import hilbert

    def _hilbert_q(i_sig):
        return SIGNAL_CENTER + hilbert(i_sig - SIGNAL_CENTER).imag

    q0 = _hilbert_q(i0)
    q1 = _hilbert_q(i1)

    data = np.stack(
        [np.stack([i0, q0], axis=0), np.stack([i1, q1], axis=0)], axis=0
    )  # (2, 2, N_amp)

    ds_16b = build_joint_stream_analysis_ds(
        coords={
            "control_state": (np.array([0, 1], dtype=int), "control state", ""),
            "analysis_axis": (np.array([0, 1], dtype=int), "analysis quadrature", ""),
            "exchange_amplitude": (amplitudes, "barrier gate voltage", "V"),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )
    _, fits = fit_16b_amplitude(
        ds_16b,
        [QP_STUB],
        exchange_duration=FIXED_DURATION_NS,
        quadrature_signal_center=SIGNAL_CENTER,
    )
    return float(fits["q1_q2"]["optimal_amplitude"])


# ── Main test ──────────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_18c_geometric_cz_phase_error_amplification_analysis(
    ld_device,
    analysis_runner,
    save_analysis_plot,
):
    """virtual_qpu N-gate × analysis-phase sweep for both target and control qubits.

    1. 16b pre-flight to find the true CZ amplitude V*.
    2. Simulate both Ramsey sweeps (target qubit phase / control qubit phase).
    3. Assemble ds_raw with both sweep variables.
    4. Run the 18c analysis and assert outputs.
    5. Physics check: ⟨|D(N,θ)|⟩_θ is flat for both sweeps at V*.
    6. Verify update_state wrote phase compensations to the CZ macro.
    """
    device = ld_device

    amplitudes = AMP_MIN + AMP_STEP * np.arange(
        int(np.round((AMP_MAX - AMP_MIN) / AMP_STEP)), dtype=np.float64
    )

    psi0 = device.ground_state()

    num_cphase_gates = np.arange(2, MAX_NUM_CPHASE_GATES + 1, 2, dtype=np.int64)
    n_gates = len(num_cphase_gates)

    # ── Step 1: locate V* from 16b-style sweep ──────────────────────────────
    v_star = _run_16b_preflight(device, amplitudes)
    assert np.isfinite(v_star) and AMP_MIN <= v_star <= AMP_MAX, (
        f"16b pre-fit returned invalid amplitude: {v_star}"
    )

    # ── Step 2: simulate target qubit phase sweep ────────────────────────────
    # cond=0: X90(target)|00⟩   cond=1: X90(target)·Xπ(control)|00⟩
    psi_tgt_cond0 = _U_X90_Q0 @ psi0
    psi_tgt_cond1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    data_target = _sweep_n_gates_phase(
        device, psi_tgt_cond0, psi_tgt_cond1, v_star, num_cphase_gates,
        _OBS_TARGET, noise_std=NOISE_STD, seed=200,
    )  # (2, n_gates, N_PHASES)

    # ── Step 3: simulate control qubit phase sweep ───────────────────────────
    # cond=0: X90(control)|00⟩   cond=1: X90(control)·Xπ(target)|00⟩
    psi_ctrl_cond0 = _U_X90_Q1 @ psi0
    psi_ctrl_cond1 = _U_X90_Q1 @ (_U_PI_Q0 @ psi0)

    data_control = _sweep_n_gates_phase(
        device, psi_ctrl_cond0, psi_ctrl_cond1, v_star, num_cphase_gates,
        _OBS_CONTROL, noise_std=NOISE_STD, seed=300,
    )  # (2, n_gates, N_PHASES)

    assert data_target.shape == (2, n_gates, N_PHASES)
    assert data_control.shape == (2, n_gates, N_PHASES)

    # ── Step 4: assemble ds_raw ──────────────────────────────────────────────
    sweep_coords = {
        "control_state": (np.array([0, 1], dtype=int), "conditional state", ""),
        "num_cphase_gates": (num_cphase_gates, "number of CZ repetitions", ""),
        "analysis_phase": (_PHASE_RADS, "analysis phase", "rad"),
    }

    # Target sweep — standard variable name used by analyse_data.
    ds_raw = build_joint_stream_analysis_ds(
        coords=sweep_coords,
        signal_per_qubit={"q1_q2": data_target},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    # Control sweep — add as E_p2_given_p1_0_ctrl_q1_q2 (already post-processed).
    import xarray as xr
    analysis_signal = "E_p2_given_p1_0"
    ds_raw[f"{analysis_signal}_ctrl_q1_q2"] = xr.DataArray(
        data_control,
        dims=["control_state", "num_cphase_gates", "analysis_phase"],
        coords={
            "control_state": ds_raw.coords["control_state"],
            "num_cphase_gates": ds_raw.coords["num_cphase_gates"],
            "analysis_phase": ds_raw.coords["analysis_phase"],
        },
    )

    # ── Step 5: run analysis ─────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "exchange_amplitude_center": float(v_star),
            "max_num_cphase_gates": int(MAX_NUM_CPHASE_GATES),
            "num_phases": N_PHASES,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    # ── Step 6: assert fit_results structure ─────────────────────────────────
    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]

    # Target sweep fields
    for key in ("peak_num_gates", "max_mean_abs_diff", "mean_contrast",
                "success_target", "phase_compensation_target"):
        assert key in fit, f"Missing key '{key}' in fit_results"

    # Control sweep fields
    for key in ("peak_num_gates_ctrl", "max_mean_abs_diff_ctrl", "mean_contrast_ctrl",
                "success_control", "phase_compensation_control"):
        assert key in fit, f"Missing key '{key}' in fit_results"

    assert "success" in fit
    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "mean_abs_diff_q1_q2" in ds_fit.data_vars
    assert "mean_abs_diff_ctrl_q1_q2" in ds_fit.data_vars
    assert "figure" in node.results

    # ── Step 7: physics checks ───────────────────────────────────────────────
    # At V* both sweeps should have flat high contrast across all even N.

    diff_target = data_target[1] - data_target[0]   # (n_gates, N_PHASES) S₁−S₀
    diff_control = data_control[1] - data_control[0]

    for diff_2d, label in [(diff_target, "target"), (diff_control, "control")]:
        mean_abs_diff = np.mean(np.abs(diff_2d), axis=-1)
        mean_mad = float(np.mean(mean_abs_diff))
        std_mad = float(np.std(mean_abs_diff))

        assert mean_mad > 0.2, (
            f"{label} sweep: mean ⟨|D|⟩_θ = {mean_mad:.4f} too low at V*={v_star:.4f} V."
        )
        assert std_mad / max(mean_mad, 1e-6) < 0.7, (
            f"{label} sweep: ⟨|D|⟩_θ varies too much (std/mean = {std_mad/mean_mad:.2f}). "
            f"Expected reasonable contrast at V*={v_star:.4f} V."
        )

    # Phase compensation values should be finite when the sweep succeeded.
    if fit["success_target"]:
        comp_t = fit["phase_compensation_target"]
        assert np.isfinite(comp_t), "phase_compensation_target should be finite"

    if fit["success_control"]:
        comp_c = fit["phase_compensation_control"]
        assert np.isfinite(comp_c), "phase_compensation_control should be finite"

    # ── Step 8: verify update_state wrote to the CZ macro ───────────────────
    qubit_pair = node.namespace["qubit_pairs"][0]
    cz_macro = qubit_pair.macros.get("cz")

    if cz_macro is not None:
        if fit["success_target"] and np.isfinite(fit["phase_compensation_target"]):
            assert np.isfinite(cz_macro.phase_shift_target), (
                "phase_shift_target should have been updated by update_state"
            )
        if fit["success_control"] and np.isfinite(fit["phase_compensation_control"]):
            assert np.isfinite(cz_macro.phase_shift_control), (
                "phase_shift_control should have been updated by update_state"
            )

    # ── Diagnostic plot ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for row_idx, (diff_2d, sweep_label) in enumerate(
        [(diff_target, "target"), (diff_control, "control")]
    ):
        mean_abs_diff_row = np.mean(np.abs(diff_2d), axis=-1)

        for col_idx, (data_2d, title) in enumerate([
            (diff_2d, f"D ({sweep_label} sweep)"),
            (np.mean(np.abs(diff_2d), axis=-1, keepdims=True).repeat(N_PHASES, axis=-1),
             f"⟨|D|⟩_θ ({sweep_label})"),
        ]):
            pass  # handled below

        axes[row_idx, 0].imshow(
            diff_2d, origin="lower", aspect="auto",
            extent=[_PHASE_RADS[0], _PHASE_RADS[-1],
                    float(num_cphase_gates[0]), float(num_cphase_gates[-1])],
            cmap="RdBu_r", interpolation="nearest",
        )
        axes[row_idx, 0].set_title(f"D = S₁−S₀ ({sweep_label} sweep)", fontsize=9)
        axes[row_idx, 0].set_xlabel("Analysis phase (rad)")
        axes[row_idx, 0].set_ylabel("N CZ gates")

        axes[row_idx, 1].plot(
            num_cphase_gates, mean_abs_diff_row, "-o", ms=5, color=f"C{row_idx}", lw=1.5,
            label="⟨|D|⟩_θ",
        )
        axes[row_idx, 1].axhline(2 / np.pi, color="red", ls="--", lw=1.2,
                                 label=f"ideal ≈ {2/np.pi:.3f}")
        axes[row_idx, 1].set_title(f"⟨|D|⟩_θ vs N ({sweep_label})", fontsize=9)
        axes[row_idx, 1].set_xlabel("N CZ gates")
        axes[row_idx, 1].set_ylabel("⟨|D(N,θ)|⟩_θ")
        axes[row_idx, 1].legend(fontsize=7)

        peak_key = "peak_num_gates" if sweep_label == "target" else "peak_num_gates_ctrl"
        comp_key = "phase_compensation_target" if sweep_label == "target" else "phase_compensation_control"
        peak_n = fit.get(peak_key, num_cphase_gates[0])
        comp = fit.get(comp_key, float("nan"))

        peak_idx = int(np.argmin(np.abs(num_cphase_gates - peak_n)))
        axes[row_idx, 2].plot(_PHASE_RADS, diff_2d[peak_idx], "-k", lw=1.5,
                              label=f"D at N*={peak_n}")
        if np.isfinite(comp):
            phi_acc = (-comp * 2 * np.pi) % (2 * np.pi)
            axes[row_idx, 2].axvline(
                phi_acc, color="orange", ls=":", lw=1.4,
                label=f"φ_acc={phi_acc:.2f}\ncomp={comp:.4f}",
            )
        axes[row_idx, 2].axhline(0, color="0.5", ls=":", lw=0.8)
        axes[row_idx, 2].set_title(f"D phase cut at N*={peak_n} ({sweep_label})", fontsize=9)
        axes[row_idx, 2].set_xlabel("Analysis phase (rad)")
        axes[row_idx, 2].legend(fontsize=7)
        xticks = np.linspace(0, 2 * np.pi, 5)
        axes[row_idx, 2].set_xticks(xticks)
        axes[row_idx, 2].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

    fig.suptitle(
        f"18c diagnostic — V*={v_star:.4f} V\n"
        "Phase compensation: "
        f"target={fit.get('phase_compensation_target', float('nan')):.4f} turns, "
        f"control={fit.get('phase_compensation_control', float('nan')):.4f} turns",
        fontsize=11,
    )
    fig.tight_layout()

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    diag_path = artifacts_dir / "diagnostics.png"
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    assert diag_path.exists()