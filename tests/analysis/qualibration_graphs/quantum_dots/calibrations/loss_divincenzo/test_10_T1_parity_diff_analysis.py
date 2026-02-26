"""Analysis test for 10_T1_parity_diff (T₁ relaxation measurement).

Uses virtual_qpu to simulate a T₁ experiment: apply a π pulse to
excite the qubit, then wait for variable idle time τ and measure the
parity difference.  The excited-state population decays exponentially
as P(τ) = offset + A·exp(−τ/T₁).

A profiled differential-evolution fit extracts T₁, amplitude, and
offset.  The π pulse amplitude is derived from the session-scoped
``calibrated_pi_half_amp`` fixture (π amp = 2× π/2 amp).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule

from .conftest import (
    DEFAULT_PULSE_DURATION_NS,
    build_parity_ds_raw,
    simulate_sweep,
)

NODE_NAME = "10_T1_parity_diff"

# ── Simulation parameters ────────────────────────────────────────────────────
PI_PULSE_DUR = DEFAULT_PULSE_DURATION_NS  # ns
TAU_MIN_NS = 16
TAU_MAX_NS = 4000  # well beyond T1 = 1000 ns for clear decay
TAU_STEP_NS = 40

# Qubits to analyse: Q1 from virtual_qpu, Q2 from synthetic exponential
MULTI_QUBITS = ["Q1", "Q2"]

# Synthetic Q2 parameters (different T1 for visual diversity)
Q2_T1_NS = 500.0
Q2_AMPLITUDE = 0.75
Q2_OFFSET = 0.05


def _simulate_q1_decay(device, pi_amp, tau_values_ns):
    """Run virtual_qpu T₁ simulation for Q1."""
    qubit_freq_ghz = device.params.qubit_freqs[0]

    def make_t1_schedule(idle_time):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=PI_PULSE_DUR,
                amplitude=pi_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_PULSE_DUR / 5,
            ),
            channel="drive_q0",
        )
        sched.play(
            SquarePulse(
                duration=idle_time,
                amplitude=0.0,
                frequency=0.0,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    result = simulate_sweep(
        device,
        make_t1_schedule,
        tsave=lambda idle_time, **_: jnp.array([0.0, PI_PULSE_DUR + idle_time], dtype=jnp.float32),
        idle_time=jnp.array(tau_values_ns, dtype=jnp.float32),
    )
    return result[..., -1]


def _synthetic_q2_decay(tau_values_ns, seed=123):
    """Generate a synthetic T₁ decay for Q2 with shorter T₁."""
    rng = np.random.default_rng(seed)
    signal = Q2_OFFSET + Q2_AMPLITUDE * np.exp(-tau_values_ns / Q2_T1_NS)
    noise = rng.normal(0, 0.08, size=signal.shape)
    return np.clip(signal + noise, 0.0, 1.0)


@pytest.mark.analysis
def test_10_T1_parity_diff_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """T₁ decay fit from virtual_qpu simulation (2 qubits)."""
    device = ld_device
    pi_amp = 2.0 * calibrated_pi_half_amp

    tau_values_ns = np.arange(TAU_MIN_NS, TAU_MAX_NS, TAU_STEP_NS)

    # ── Simulate / generate data for both qubits ─────────────────────
    pdiff_q1 = _simulate_q1_decay(device, pi_amp, tau_values_ns)
    pdiff_q2 = _synthetic_q2_decay(tau_values_ns)

    assert pdiff_q1.shape == (len(tau_values_ns),)
    assert np.max(pdiff_q1) > 0.01, "Q1 simulation should show some signal"

    # ── Build ds_raw (1D: tau, 2 qubits) ─────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "tau": (tau_values_ns.astype(float), "idle time", "ns"),
        },
        pdiff_per_qubit={"Q1": pdiff_q1, "Q2": pdiff_q2},
    )

    # ── Run analysis ──────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "tau_min": TAU_MIN_NS,
            "tau_max": TAU_MAX_NS,
            "tau_step": TAU_STEP_NS,
        },
        analyse_qubits=MULTI_QUBITS,
    )

    # ── Assertions: Q1 (virtual_qpu, T1 ≈ 1000 ns) ──────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Q1 analysis should succeed: {fit_q1}"

    t1_q1 = fit_q1["T1"]
    assert 200 < t1_q1 < 5000, f"Q1 T1 should be near 1000 ns, got {t1_q1:.1f} ns"
    assert fit_q1["amplitude"] > 0.01, f"Q1 expected positive amplitude, got {fit_q1['amplitude']}"

    gamma_q1 = fit_q1["decay_rate"]
    assert np.isfinite(gamma_q1) and gamma_q1 > 0
    assert abs(1.0 / gamma_q1 - t1_q1) < 1e-6, "decay_rate should be 1/T1"

    # ── Assertions: Q2 (synthetic, T1 ≈ 500 ns) ──────────────────────
    fit_q2 = node.results["fit_results"]["Q2"]
    assert fit_q2["success"], f"Q2 analysis should succeed: {fit_q2}"

    t1_q2 = fit_q2["T1"]
    assert 100 < t1_q2 < 2000, f"Q2 T1 should be near 500 ns, got {t1_q2:.1f} ns"
    assert fit_q2["amplitude"] > 0.01, f"Q2 expected positive amplitude, got {fit_q2['amplitude']}"
