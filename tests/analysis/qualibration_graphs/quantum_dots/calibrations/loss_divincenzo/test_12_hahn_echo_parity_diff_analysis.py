"""Analysis test for 12_hahn_echo_parity_diff.

Uses virtual_qpu to simulate a Hahn echo (spin echo) decay for a
Loss-DiVincenzo spin qubit.  The echo sequence is

    π/2 – τ – π – τ – π/2

and the parity-difference signal decays as  P(τ) = offset + A·exp(−2τ/T₂).
The π/2 pulse amplitude is pre-calibrated via a quick power-Rabi sweep
(``calibrated_pi_half_amp`` fixture in conftest).

Two qubits are tested:
  - Q1: full virtual_qpu physics simulation (T₂ = 400 ns from device config).
  - Q2: synthetic exponential decay with T₂ = 600 ns for multi-qubit coverage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule

from .conftest import (
    ANALYSE_QUBITS,
    ARTIFACTS_BASE,
    DEFAULT_PULSE_DURATION_NS,
    build_parity_ds_raw,
    simulate_sweep,
)

NODE_NAME = "12_hahn_echo_parity_diff"

# ── Simulation parameters ────────────────────────────────────────────────────
PI_HALF_DUR = DEFAULT_PULSE_DURATION_NS  # ns
MAX_TAU_NS = 600
N_TAU_POINTS = 80
NOISE_STD = 0.03

# Q2 synthetic parameters
Q2_T2_ECHO_NS = 600.0
Q2_AMPLITUDE = 0.65
Q2_OFFSET = 0.05


@pytest.mark.analysis
def test_12_hahn_echo_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """Hahn echo with pre-calibrated pulses: fit T₂_echo for two qubits."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_half_amp = calibrated_pi_half_amp
    pi_amp = 2.0 * pi_half_amp

    # ── Sweep axis ────────────────────────────────────────────────────────
    tau_values = jnp.linspace(16, MAX_TAU_NS, N_TAU_POINTS, dtype=jnp.float32)

    # ── Hahn echo schedule factory ────────────────────────────────────────
    def _build_hahn_schedule(tau):
        """Build (but do not resolve) a Hahn echo Schedule for a given τ."""
        sched = Schedule()
        # First π/2 pulse
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        # First idle period τ
        sched.play(
            SquarePulse(duration=tau, amplitude=0.0, frequency=0.0),
            channel="drive_q0",
        )
        # Refocusing π pulse
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        # Second idle period τ
        sched.play(
            SquarePulse(duration=tau, amplitude=0.0, frequency=0.0),
            channel="drive_q0",
        )
        # Final π/2 pulse
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        return sched

    def make_hahn_schedule(tau):
        return _build_hahn_schedule(tau).resolve()

    # ── Save schedule plot for the first sweep point ──────────────────────
    first_tau = float(tau_values[0])
    sched_first = _build_hahn_schedule(first_tau)
    ax = sched_first.plot()
    ax.set_title(f"Hahn echo schedule (τ = {first_tau:.0f} ns)")
    fig_sched = ax.get_figure()
    fig_sched.tight_layout()
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fig_sched.savefig(artifacts_dir / "schedule.png", dpi=200)
    plt.close(fig_sched)

    # ── Simulate Q1 (1D sweep over τ) ────────────────────────────────────
    result = simulate_sweep(
        device,
        make_hahn_schedule,
        tsave=lambda tau, **_: jnp.array([0.0, 3 * PI_HALF_DUR + 2 * tau]),
        noise_std=NOISE_STD,
        tau=tau_values,
    )
    # result shape: (n_tau, n_tsave) → take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.01, "Simulation should show some signal"

    # ── Synthetic Q2 data ─────────────────────────────────────────────────
    tau_np = np.asarray(tau_values, dtype=np.float64)
    rng = np.random.default_rng(seed=99)
    pdiff_q2 = Q2_OFFSET + Q2_AMPLITUDE * np.exp(-2.0 * tau_np / Q2_T2_ECHO_NS)
    pdiff_q2 += rng.normal(0, NOISE_STD, size=pdiff_q2.shape)
    pdiff_q2 = np.clip(pdiff_q2, 0.0, 1.0)

    # ── Build ds_raw ──────────────────────────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={"tau": (tau_np, "per-arm idle time", "ns")},
        pdiff_per_qubit={"Q1": pdiff_q1, "Q2": pdiff_q2},
    )

    # ── Run analysis ──────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "tau_min": 16,
            "tau_max": int(MAX_TAU_NS),
            "tau_step": 16,
        },
        analyse_qubits=["Q1", "Q2"],
    )

    # ── Assertions ────────────────────────────────────────────────────────
    assert "fit_results" in node.results

    # Q1: physics simulation (T₂ = 400 ns from device config)
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Q1 analysis should succeed: {fit_q1}"
    assert fit_q1["T2_echo"] > 0, f"Q1 T2_echo should be positive, got {fit_q1['T2_echo']}"
    assert np.isfinite(fit_q1["T2_echo"]), f"Q1 T2_echo should be finite, got {fit_q1['T2_echo']}"
    assert fit_q1["decay_rate"] > 0, f"Q1 decay_rate should be positive, got {fit_q1['decay_rate']}"

    # Q2: synthetic data (T₂_echo = 600 ns)
    fit_q2 = node.results["fit_results"]["Q2"]
    assert fit_q2["success"], f"Q2 analysis should succeed: {fit_q2}"
    assert abs(fit_q2["T2_echo"] - Q2_T2_ECHO_NS) < 0.3 * Q2_T2_ECHO_NS, (
        f"Q2 T2_echo should be near {Q2_T2_ECHO_NS} ns, got {fit_q2['T2_echo']:.1f} ns"
    )
    assert abs(fit_q2["amplitude"] - Q2_AMPLITUDE) < 0.3 * Q2_AMPLITUDE, (
        f"Q2 amplitude should be near {Q2_AMPLITUDE}, got {fit_q2['amplitude']:.4f}"
    )
    assert abs(fit_q2["offset"] - Q2_OFFSET) < 0.1, (
        f"Q2 offset should be near {Q2_OFFSET}, got {fit_q2['offset']:.4f}"
    )
