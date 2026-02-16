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
    ANALYSE_QUBITS,
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


@pytest.mark.analysis
def test_10_T1_parity_diff_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """T₁ decay fit from virtual_qpu simulation."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_amp = 2.0 * calibrated_pi_half_amp  # full π amplitude

    # ── Idle-time sweep values ────────────────────────────────────────
    tau_values_ns = np.arange(TAU_MIN_NS, TAU_MAX_NS, TAU_STEP_NS)

    # ── Simulate T1 decay ─────────────────────────────────────────────
    def make_t1_schedule(idle_time):
        sched = Schedule()
        # π pulse to excite qubit
        sched.play(
            GaussianIQPulse(
                duration=PI_PULSE_DUR,
                amplitude=pi_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_PULSE_DUR / 5,
            ),
            channel="drive_q0",
        )
        # Variable idle time
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
    pdiff_1d = result[..., -1]  # (n_tau,)

    assert pdiff_1d.shape == (len(tau_values_ns),)
    assert np.max(pdiff_1d) > 0.01, "Simulation should show some signal"

    # ── Build ds_raw (1D: tau) ────────────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "tau": (tau_values_ns.astype(float), "idle time", "ns"),
        },
        pdiff_per_qubit={q: pdiff_1d for q in ANALYSE_QUBITS},
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
    )

    # ── Assertions ────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # T1 should be close to the simulated value (1000 ns)
    t1_ns = fit_q1["T1"]
    assert 200 < t1_ns < 5000, f"T1 should be near 1000 ns, got {t1_ns:.1f} ns"

    # Amplitude should be positive (decay from excited state)
    amp = fit_q1["amplitude"]
    assert amp > 0.01, f"Expected positive decay amplitude, got {amp}"

    # Decay rate should be positive and consistent with T1
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma) and gamma > 0, f"Expected finite gamma > 0, got {gamma}"
    assert abs(1.0 / gamma - t1_ns) < 1e-6, "decay_rate should be 1/T1"
