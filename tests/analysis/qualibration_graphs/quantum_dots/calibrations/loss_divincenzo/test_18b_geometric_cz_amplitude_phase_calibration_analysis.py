"""Analysis test for 18b_geometric_cz_amplitude_phase_calibration.

Uses virtual_qpu (Loss-DiVincenzo Hamiltonian + dynamiqs) for the exchange
interaction and perfect unitary gates (matrix multiplication) for state
preparation and the analysis rotation.

Simulation strategy
-------------------
State preparation (X90 on target, π on control for ctrl=|1⟩) and the closing
π/2 analysis rotation are applied as exact 4×4 unitary matrices — no ODE
solve is needed for these ideal gates.  Only the exchange interaction is
time-evolved through dynamiqs.

For each of the N_PHASES analysis phases θ ∈ [0, 2π) we fold the closing
rotation U(θ) into a phase-dependent parity observable:

    P_obs(θ) = (U(θ) ⊗ I₂)† P_odd (U(θ) ⊗ I₂)

where U(θ) = (1/√2) [[1, −i e^{−iθ}], [−i e^{iθ}, 1]] rotates the target
qubit around the in-plane axis n = (cos θ, sin θ, 0) by π/2.  The N_PHASES
observables are stacked into a (N_PHASES, 4, 4) tensor so a single
jax.vmap pass over exchange amplitudes returns (N_amp, N_phases) signals
per control state.

Test flow
---------
1. Run a 16b-style amplitude sweep (at fixed duration) to locate the true
   CZ amplitude V* where the conditional phase reaches π.
    2. Run the 18b 2-D amplitude × phase sweep centred on the same amplitude
   range.
3. Assemble ds_raw with dims (control_state=2, exchange_amplitude, analysis_phase).
4. Run the 18b analysis (analysis_runner) and check results.

Analysis metric
---------------
The analysis uses the mean absolute parity difference:

    mean_abs_diff(V) = ⟨|D(V, θ)|⟩_θ  ∝  |sin(δ(V)/2)|

where D = S(ctrl=|1⟩) − S(ctrl=|0⟩).  This is MAXIMISED at the CZ
amplitude V* where the conditional phase δ = π.  Using the plain mean ⟨D⟩_θ
is not suitable because D is a pure sinusoid in θ and the phases are sampled
uniformly over [0, 2π), giving ⟨D⟩_θ ≈ 0 for all amplitudes.
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

NODE_NAME = "18b_geometric_cz_amplitude_phase_calibration"
QP_STUB = SimpleNamespace(name="q1_q2")

FIXED_DURATION_NS = 50.0
RETURN_TO_INIT_NS = 16.0

AMP_MIN = 0.05
AMP_MAX = 0.40
AMP_STEP = 0.005
N_AMPS = int(np.round((AMP_MAX - AMP_MIN) / AMP_STEP))

N_PHASES = 21
NOISE_STD = 0.15
SIGNAL_CENTER = 0.5

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=1_000_000)}

# ── Perfect single-qubit unitaries in the 4-qubit Hilbert space ──────────

_I2 = jnp.eye(2, dtype=jnp.complex64)

_Rx_half_pi = jnp.array(
    [[1.0, -1j], [-1j, 1.0]], dtype=jnp.complex64
) / jnp.sqrt(2.0)

_Rx_pi = jnp.array(
    [[0.0, -1j], [-1j, 0.0]], dtype=jnp.complex64
)

# X90 on target (mode 0): Rx(π/2) ⊗ I₂
_U_X90_Q0 = jnp.kron(_Rx_half_pi, _I2)

# π on control (mode 1): I₂ ⊗ Rx(π)
_U_PI_Q1 = jnp.kron(_I2, _Rx_pi)

# Two-qubit parity projector P_odd = |01⟩⟨01| + |10⟩⟨10|
_P_ODD = jnp.array(
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    dtype=jnp.complex64,
)


# ── Phase-dependent observable ─────────────────────────────────────────────


def _make_phase_obs(
    theta_rad: float, control_closing: jnp.ndarray = _I2
) -> jnp.ndarray:
    """P_obs(θ) = U_close† P_odd U_close for the PSB parity readout.

    The closing readout maps onto the two-qubit *parity* projector P_odd.
    The closing unitary is

        U_close(θ) = (I₂ ⊗ control_closing) · (U_t(θ) ⊗ I₂)
                   =  U_t(θ) ⊗ control_closing

    where U_t(θ) = (1/√2) [[1, −i e^{−iθ}], [−i e^{iθ}, 1]] is the π/2
    rotation around the in-plane axis n = (cos θ, sin θ, 0) — the effect of
    frame_rotation_2pi(θ/2π) followed by X90 on the target.

    ``control_closing`` is the closing operation applied to the *control*
    qubit just before the parity readout:

      * ctrl=|0⟩ branch: identity (no closing control pulse).
      * ctrl=|1⟩ branch: Rx(π) — the "undo the partner spin flip" X180 the
        real node plays so PSB parity maps back onto the target qubit
        (``18b...py:243-249``).
    """
    U_t = jnp.array(
        [
            [1.0, -1j * jnp.exp(-1j * theta_rad)],
            [-1j * jnp.exp(1j * theta_rad), 1.0],
        ],
        dtype=jnp.complex64,
    ) / jnp.sqrt(2.0)
    U_2q = jnp.kron(U_t, control_closing)  # target ⊗ control basis
    return U_2q.conj().T @ _P_ODD @ U_2q


# Pre-compute and stack all N_PHASES observables into (N_PHASES, 4, 4).
# Evaluated once in Python; treated as a constant inside vmap.
_PHASE_FRACS = np.linspace(0, 1, N_PHASES, endpoint=False)

# ctrl=|0⟩: no closing control pulse.
_OBS_CTRL0 = jnp.stack(
    [_make_phase_obs(float(th * 2 * np.pi), _I2) for th in _PHASE_FRACS],
    axis=0,
)  # (N_PHASES, 4, 4)

# ctrl=|1⟩: closing X180 on the control qubit before the parity readout —
# this is the full PSB parity-readout mechanism the real node applies.
_OBS_CTRL1 = jnp.stack(
    [_make_phase_obs(float(th * 2 * np.pi), _Rx_pi) for th in _PHASE_FRACS],
    axis=0,
)  # (N_PHASES, 4, 4)


# ── Exchange-only schedule ─────────────────────────────────────────────────


def _make_exchange_schedule(exchange_amplitude) -> object:
    """Exchange pulse at fixed duration + return-to-init idle."""
    sched = Schedule()
    ref_ex = sched.play(
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
        after=[ref_ex],
    )
    return sched.resolve()


_TSAVE = jnp.array(
    [0.0, FIXED_DURATION_NS + RETURN_TO_INIT_NS], dtype=jnp.float32
)


# ── Core simulation helpers ────────────────────────────────────────────────


def _sweep_amplitude_phase(
    device,
    psi_init: jnp.ndarray,
    amplitudes: jnp.ndarray,
    *,
    obs: jnp.ndarray,
    noise_std: float = NOISE_STD,
    seed: int = 42,
) -> np.ndarray:
    """For each exchange amplitude, compute the parity signal for all N_PHASES.

    The analysis phase sweep is implemented analytically: the closing readout
    (target π/2 + optional control X180) is folded into the phase-dependent
    parity observables ``obs`` and evaluated via batched einsum — no extra ODE
    solves are needed.

    Parameters
    ----------
    obs : jnp.ndarray, shape (N_PHASES, 4, 4)
        Per-phase parity observables for this control branch (``_OBS_CTRL0``
        or ``_OBS_CTRL1``).

    Returns
    -------
    np.ndarray, shape (N_amp, N_PHASES)
        Parity signal E[p2|p1=0] for each (amplitude, analysis_phase) pair.
    """

    def _inner(exchange_amplitude):
        sched = _make_exchange_schedule(exchange_amplitude)
        H_t = device.hamiltonian(sched)
        sol = vqpu_simulate(
            H_t, psi_init, _TSAVE, solver="se", options=_SOLVER_KW
        )
        # Final state: dynamiqs ket → (4, 1) → squeeze to (4,)
        raw = sol.states[-1]
        psi = jnp.squeeze(raw.to_jax() if hasattr(raw, "to_jax") else jnp.asarray(raw))
        # Expectation for each phase:
        # obs @ psi → (N_PHASES, 4); then psi† · each row
        obs_psi = jnp.einsum("kij,j->ki", obs, psi)  # (N_PHASES, 4)
        return jnp.real(jnp.einsum("i,ki->k", jnp.conj(psi), obs_psi))  # (N_PHASES,)

    result = np.asarray(
        vqpu_sweep(_inner, mode="outer", exchange_amplitude=amplitudes)
    )  # (N_amp, N_PHASES)

    if noise_std > 0:
        rng = np.random.default_rng(seed=seed)
        result = result + rng.normal(0, noise_std, size=result.shape)
        result = np.clip(result, 0.0, 1.0)
    return result


# ── 16b pre-fit: locate true CZ amplitude ─────────────────────────────────


def _run_16b_preflight(
    device,
    amplitudes: np.ndarray,
) -> float:
    """16b-style amplitude sweep to locate the π conditional-phase amplitude.

    State preparation and readout use perfect unitary gates; only the
    exchange pulse is time-evolved.  Returns the fitted optimal amplitude.
    """
    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    _obs_x90 = dq.asqarray(_U_X90_Q0.conj().T @ _P_ODD @ _U_X90_Q0)

    def _sweep_ctrl(psi_init, seed):
        def _inner(exchange_amplitude):
            sched = _make_exchange_schedule(exchange_amplitude)
            H_t = device.hamiltonian(sched)
            sol = vqpu_simulate(
                H_t, psi_init, _TSAVE, solver="se", options=_SOLVER_KW
            )
            return vqpu_expval(sol.states, _obs_x90)

        result = np.asarray(
            vqpu_sweep(_inner, mode="outer", exchange_amplitude=amplitudes)
        )
        rng = np.random.default_rng(seed=seed)
        return np.clip(result + rng.normal(0, 0.03, size=result.shape), 0.0, 1.0)

    r0 = _sweep_ctrl(psi_ctrl0, seed=10)
    r1 = _sweep_ctrl(psi_ctrl1, seed=11)

    i0 = np.asarray(r0[..., -1], dtype=np.float64)
    i1 = np.asarray(r1[..., -1], dtype=np.float64)

    # Synthesise Q quadrature via Hilbert transform (matching 16b approach)
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


# ── Main test ──────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_18b_geometric_cz_amplitude_phase_analysis(
    ld_device,
    analysis_runner,
    save_analysis_plot,
):
    """virtual_qpu amplitude × analysis-phase sweep for 18b.

    1. Run a 16b-style pre-fit to locate the true CZ amplitude V*.
    2. Build the 2-D (amplitude × phase) dataset using the exchange dynamics
       and folded phase-dependent observables.
    3. Run the 18b analysis and assert basic sanity on the output.
    4. Emit diagnostic plots: raw 2-D heatmaps, D(V,θ), and ⟨|D|⟩_θ vs V.
    5. Check that argmax(⟨|D|⟩_θ) agrees with V*(18b) to within 5 steps.
    """
    device = ld_device

    amplitudes = AMP_MIN + AMP_STEP * np.arange(N_AMPS, dtype=np.float64)
    amplitudes_j = jnp.asarray(amplitudes, dtype=jnp.float32)
    phases_rad = _PHASE_FRACS * 2 * np.pi

    psi0 = device.ground_state()
    psi_ctrl0 = _U_X90_Q0 @ psi0
    psi_ctrl1 = _U_X90_Q0 @ (_U_PI_Q1 @ psi0)

    # ── Step 1: locate V* from 16b-style sweep ─────────────────────────
    v_star_16b = _run_16b_preflight(device, amplitudes)
    assert np.isfinite(v_star_16b) and AMP_MIN <= v_star_16b <= AMP_MAX, (
        f"16b pre-fit returned invalid amplitude: {v_star_16b}"
    )

    # ── Step 2: 2-D amplitude × phase sweep ────────────────────────────
    sig_ctrl0 = _sweep_amplitude_phase(
        device, psi_ctrl0, amplitudes_j, obs=_OBS_CTRL0,
        noise_std=NOISE_STD, seed=100,
    )  # (N_amp, N_PHASES)
    sig_ctrl1 = _sweep_amplitude_phase(
        device, psi_ctrl1, amplitudes_j, obs=_OBS_CTRL1,
        noise_std=NOISE_STD, seed=101,
    )  # (N_amp, N_PHASES)

    assert sig_ctrl0.shape == (N_AMPS, N_PHASES)
    assert sig_ctrl1.shape == (N_AMPS, N_PHASES)

    # ── Step 3: assemble ds_raw ─────────────────────────────────────────
    # dims: (control_state=2, exchange_amplitude, analysis_phase)
    data = np.stack([sig_ctrl0, sig_ctrl1], axis=0)  # (2, N_amp, N_phases)

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "control_state": (np.array([0, 1], dtype=int), "control state", ""),
            "exchange_amplitude": (amplitudes, "barrier gate voltage", "V"),
            "analysis_phase": (phases_rad, "analysis phase", "rad"),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    # ── Step 4: run analysis ────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_exchange_amplitude": float(AMP_MIN),
            "max_exchange_amplitude": float(AMP_MAX),
            "amplitude_step": float(AMP_STEP),
            "num_phases": N_PHASES,
        },
        namespace_overrides={
            "exchange_duration": int(FIXED_DURATION_NS),
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    # ── Step 5: assert outputs ──────────────────────────────────────────
    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]
    assert "optimal_amplitude" in fit
    assert "min_mean_abs_diff" in fit
    assert "success" in fit

    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "mean_abs_diff_q1_q2" in ds_fit.data_vars, (
        f"Expected 'mean_abs_diff_q1_q2' in ds_fit; got {list(ds_fit.data_vars)}"
    )

    assert "figure" in node.results

    # ── Step 6: diagnostic plots ────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_2d = sig_ctrl1 - sig_ctrl0  # (N_amp, N_phases)
    mean_abs_diff = np.mean(np.abs(diff_2d), axis=-1)  # (N_amp,) ∝ |sin(δ/2)|

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Row 0: 2-D heatmaps
    for ax, data_2d, title in [
        (axes[0, 0], sig_ctrl0, "S(ctrl=|0⟩)"),
        (axes[0, 1], sig_ctrl1, "S(ctrl=|1⟩)"),
        (axes[0, 2], diff_2d, "D = S₁ − S₀"),
    ]:
        vmax = np.abs(data_2d).max()
        cmap = "viridis" if "ctrl" in title else "RdBu_r"
        im = ax.imshow(
            data_2d,
            origin="lower",
            aspect="auto",
            extent=[phases_rad[0], phases_rad[-1], amplitudes[0], amplitudes[-1]],
            vmin=-vmax if "D" in title else None,
            vmax=vmax if "D" in title else None,
            cmap=cmap,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax)
        ax.axhline(v_star_16b, color="red", ls="--", lw=1.2, label=f"V*(16b)={v_star_16b:.3f}V")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Analysis phase (rad)")
        ax.set_ylabel("Exchange amplitude (V)")
        ax.legend(fontsize=7)

    # Row 1: 1-D diagnostics
    axes[1, 0].plot(amplitudes, mean_abs_diff, "-", color="C0", lw=1.5, label="⟨|D|⟩_θ")
    axes[1, 0].axvline(v_star_16b, color="red", ls="--", lw=1.2, label="V*(16b)")
    if fit["success"]:
        axes[1, 0].axvline(
            fit["optimal_amplitude"], color="C1", ls="--", lw=1.2,
            label=f"V*(18b)={fit['optimal_amplitude']:.3f}V"
        )
    axes[1, 0].set_title("⟨|D|⟩_θ vs amplitude", fontsize=10)
    axes[1, 0].set_xlabel("Exchange amplitude (V)")
    axes[1, 0].set_ylabel("Mean |parity difference|")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(amplitudes, mean_abs_diff, "-", color="C2", lw=1.5,
                    label="⟨|D|⟩_θ ∝ |sin(δ/2)|")
    axes[1, 1].axvline(v_star_16b, color="red", ls="--", lw=1.2, label="V*(16b)")
    axes[1, 1].set_title("⟨|D|⟩_θ vs exchange amplitude", fontsize=10)
    axes[1, 1].set_xlabel("Exchange amplitude (V)")
    axes[1, 1].set_ylabel("⟨|D|⟩_θ")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].annotate(
        "CZ point = min (δ=π)",
        xy=(v_star_16b, mean_abs_diff[np.argmin(np.abs(amplitudes - v_star_16b))]),
        xytext=(v_star_16b + 0.03, mean_abs_diff.max() * 0.5),
        arrowprops=dict(arrowstyle="->", color="k"),
        fontsize=8,
    )

    # Phase cuts at V*
    v_star_idx = int(np.argmin(np.abs(amplitudes - v_star_16b)))
    axes[1, 2].plot(phases_rad, sig_ctrl0[v_star_idx], "-o", ms=3, color="C0",
                    lw=1.2, label="ctrl |0⟩")
    axes[1, 2].plot(phases_rad, sig_ctrl1[v_star_idx], "-s", ms=3, color="C1",
                    lw=1.2, label="ctrl |1⟩")
    axes[1, 2].plot(phases_rad, diff_2d[v_star_idx], "-", color="k", lw=1.5,
                    label="D = S₁ − S₀")
    axes[1, 2].axhline(0, color="0.5", ls=":", lw=0.8)
    axes[1, 2].set_title(f"Phase cuts at V*(16b) = {v_star_16b:.3f} V", fontsize=10)
    axes[1, 2].set_xlabel("Analysis phase (rad)")
    axes[1, 2].set_ylabel("Parity signal")
    axes[1, 2].legend(fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    axes[1, 2].set_xticks(xticks)
    axes[1, 2].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

    fig.suptitle(
        "18b diagnostic — amplitude × phase sweep\n"
        f"16b V*={v_star_16b:.4f} V  |  "
        f"⟨|D|⟩_θ is MINIMISED at CZ point (J·t/2 = π/2)",
        fontsize=11,
    )
    fig.tight_layout()

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    diag_path = artifacts_dir / "diagnostics.png"
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    assert diag_path.exists()

    # ── Step 6b: parity-readout fringe-doubling probe ───────────────────
    # Directly compare, NOISELESS, the ctrl=|1⟩ phase fringe at every
    # amplitude computed two ways:
    #   * "single-qubit readout": no closing control pulse (_OBS_CTRL0)
    #   * "PSB parity readout":   closing control X180 (_OBS_CTRL1), i.e.
    #                             the mechanism the real node uses.
    # If the parity readout folds the control qubit's phase response into the
    # signal it will show a 2θ (4π) fringe over the 2π sweep.
    sig_ctrl1_singleq = _sweep_amplitude_phase(
        device, psi_ctrl1, amplitudes_j, obs=_OBS_CTRL0, noise_std=0.0
    )
    sig_ctrl1_parity = _sweep_amplitude_phase(
        device, psi_ctrl1, amplitudes_j, obs=_OBS_CTRL1, noise_std=0.0
    )

    def _fringe_periods(signal_1d: np.ndarray) -> float:
        """Dominant number of full oscillations across the 2π phase sweep
        via the peak of the real FFT (excluding DC)."""
        centred = signal_1d - signal_1d.mean()
        spectrum = np.abs(np.fft.rfft(centred))
        spectrum[0] = 0.0
        return float(np.argmax(spectrum))

    cut_singleq = sig_ctrl1_singleq[v_star_idx]
    cut_parity = sig_ctrl1_parity[v_star_idx]
    periods_singleq = _fringe_periods(cut_singleq)
    periods_parity = _fringe_periods(cut_parity)

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    ax2[0].plot(phases_rad, cut_singleq, "-o", ms=3, color="C0",
                label=f"single-qubit readout ({periods_singleq:.0f} period)")
    ax2[0].plot(phases_rad, cut_parity, "-s", ms=3, color="C3",
                label=f"PSB parity readout ({periods_parity:.0f} period)")
    ax2[0].set_title(
        f"ctrl=|1⟩ fringe at V*={v_star_16b:.3f} V (noiseless)", fontsize=10
    )
    ax2[0].set_xlabel("Analysis phase (rad)")
    ax2[0].set_ylabel("Signal")
    ax2[0].set_xticks(xticks)
    ax2[0].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax2[0].legend(fontsize=8)

    # Periods vs amplitude — does doubling persist across the sweep?
    per_singleq = np.array([_fringe_periods(sig_ctrl1_singleq[i]) for i in range(N_AMPS)])
    per_parity = np.array([_fringe_periods(sig_ctrl1_parity[i]) for i in range(N_AMPS)])
    ax2[1].plot(amplitudes, per_singleq, "-", color="C0", label="single-qubit readout")
    ax2[1].plot(amplitudes, per_parity, "-", color="C3", label="PSB parity readout")
    ax2[1].axvline(v_star_16b, color="red", ls="--", lw=1.0, label="V*(16b)")
    ax2[1].set_title("Dominant fringe periods over 2π vs amplitude", fontsize=10)
    ax2[1].set_xlabel("Exchange amplitude (V)")
    ax2[1].set_ylabel("Periods across 2π sweep")
    ax2[1].legend(fontsize=8)
    fig2.suptitle("18b — parity-readout fringe-doubling probe", fontsize=11)
    fig2.tight_layout()
    probe_path = artifacts_dir / "parity_readout_probe.png"
    fig2.savefig(probe_path, dpi=150)
    plt.close(fig2)
    assert probe_path.exists()

    print(
        f"\n[parity-probe] ctrl=|1⟩ fringe periods at V*: "
        f"single-qubit={periods_singleq:.0f}, parity={periods_parity:.0f}"
    )

    # ── Step 7: physics sanity check ────────────────────────────────────
    # ⟨|D|⟩_θ ∝ |cos(J(V)·t/2)| — MINIMISED at the CZ point where
    # J(V*)·t/2 = π/2 (conditional phase δ = π).
    # The argmin should agree with V* from the 16b pre-fit to within a
    # few amplitude steps.
    v_star_mad = amplitudes[int(np.argmin(mean_abs_diff))]
    assert abs(v_star_mad - v_star_16b) <= 5 * AMP_STEP, (
        f"Min ⟨|D|⟩_θ at {v_star_mad:.4f} V is more than "
        f"5 steps from 16b V* = {v_star_16b:.4f} V. Physics check failed."
    )

    # The 18b analysis should also find V* near the 16b value.
    assert fit["success"], (
        f"18b analysis did not converge; fit={fit}"
    )
    assert abs(fit["optimal_amplitude"] - v_star_16b) <= 5 * AMP_STEP, (
        f"18b optimal amplitude {fit['optimal_amplitude']:.4f} V is more than "
        f"5 steps from 16b V* = {v_star_16b:.4f} V."
    )