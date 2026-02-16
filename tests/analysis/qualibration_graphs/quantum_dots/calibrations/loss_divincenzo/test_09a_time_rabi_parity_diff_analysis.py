"""Analysis test for 09a_time_rabi_parity_diff.

Uses virtual_qpu to simulate a Lindblad time-Rabi oscillation for a
Loss-DiVincenzo spin qubit with a Gaussian IQ pulse, builds a synthetic
1D ``ds_raw`` (pulse duration only), and runs the FFT-based analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse
from virtual_qpu.schedule import Schedule

from .conftest import ANALYSE_QUBITS, build_parity_ds_raw, simulate_sweep

NODE_NAME = "09a_time_rabi_parity_diff"

# ── Node-specific simulation parameters ─────────────────────────────────────
DRIVE_AMP_GHZ = 0.008
MAX_DURATION_NS = 800
N_DURATION_POINTS = 200


@pytest.mark.analysis
def test_09a_time_rabi_analysis(ld_device, analysis_runner):
    """Lindblad time-Rabi (Gaussian pulse) with 1D FFT analysis."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]

    # ── Sweep axis ───────────────────────────────────────────────────────────
    durations = jnp.linspace(4, MAX_DURATION_NS, N_DURATION_POINTS, dtype=jnp.float32)

    # ── Schedule factory (Gaussian IQ pulse, on-resonance, 1D duration sweep) ─
    def make_schedule(dur):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=dur,
                amplitude=DRIVE_AMP_GHZ,
                frequency=qubit_freq_ghz,
                sigma=dur / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    # ── Simulate (1D sweep: duration only) ───────────────────────────────────
    result = simulate_sweep(
        device,
        make_schedule,
        tsave=lambda dur, **_: jnp.array([0.0, dur]),
        dur=durations,
    )
    # result shape: (n_dur, n_tsave) → take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.05, "Simulation should show spin-flip signal"

    # ── Build ds_raw (1D: pulse_duration only) ───────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "pulse_duration": (np.asarray(durations), "qubit pulse duration", "ns"),
        },
        pdiff_per_qubit={q: pdiff_q1 for q in ANALYSE_QUBITS},
    )

    # ── Run analysis ─────────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_wait_time_in_ns": 4,
            "max_wait_time_in_ns": MAX_DURATION_NS,
            "time_step_in_ns": MAX_DURATION_NS // N_DURATION_POINTS if N_DURATION_POINTS > 1 else 4,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # π-time in reasonable range
    t_pi = fit_q1["optimal_duration"]
    assert 30 < t_pi < 500, f"Expected t_π ∈ [30, 500] ns, got {t_pi:.0f}"

    # Rabi frequency should be finite and positive
    omega = fit_q1["rabi_frequency"]
    assert np.isfinite(omega) and omega > 0, f"Expected finite Ω > 0, got {omega}"

    # Decay rate should be present and finite
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma), f"Expected finite γ, got {gamma}"

    # Damped sinusoid fit should have been used (not just FFT fallback)
    sinusoid = fit_q1.get("_sinusoid_fit")
    assert sinusoid is not None, "Damped sinusoid fit should succeed on clean simulation data"
    assert sinusoid["frequency"] > 0, "Fitted frequency should be positive"
    assert sinusoid["gamma"] >= 0, "Fitted decay rate should be non-negative"
    # Consistency: sinusoid frequency should agree with reported Rabi values
    omega_from_sin = 2.0 * np.pi * sinusoid["frequency"]
    assert (
        abs(omega_from_sin - omega) / omega < 0.01
    ), f"Sinusoid freq and reported Ω should agree: {omega_from_sin:.5f} vs {omega:.5f}"
