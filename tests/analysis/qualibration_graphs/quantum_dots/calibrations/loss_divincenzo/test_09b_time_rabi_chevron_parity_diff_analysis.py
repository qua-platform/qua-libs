"""Analysis test for 09b_time_rabi_chevron_parity_diff.

Uses virtual_qpu to simulate a Lindblad time-Rabi chevron for a
Loss-DiVincenzo spin qubit with a Gaussian IQ pulse, builds a synthetic
``ds_raw``, and runs the FFT-based analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse
from virtual_qpu.schedule import Schedule

from .conftest import ANALYSE_QUBITS, build_parity_ds_raw, simulate_sweep

NODE_NAME = "09b_time_rabi_chevron_parity_diff"

# ── Node-specific simulation parameters ─────────────────────────────────────
DRIVE_AMP_GHZ = 0.008
MAX_DURATION_NS = 800
N_DURATION_POINTS = 200
FREQ_SPAN_MHZ = 100.0
FREQ_STEP_MHZ = 1.0


@pytest.mark.analysis
def test_09b_time_rabi_chevron_analysis(ld_device, analysis_runner):
    """Lindblad time-Rabi chevron (Gaussian pulse) with FFT analysis."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]

    # ── Sweep axes ───────────────────────────────────────────────────────────
    durations = jnp.linspace(4, MAX_DURATION_NS, N_DURATION_POINTS, dtype=jnp.float32)
    # Only need initial and final state for each pulse duration
    tsave = jnp.array([0.0, 1.0], dtype=jnp.float32)  # placeholder; overridden per pulse

    span = FREQ_SPAN_MHZ * 1e-3
    step = FREQ_STEP_MHZ * 1e-3
    drive_freqs = jnp.arange(
        qubit_freq_ghz - span / 2,
        qubit_freq_ghz + span / 2,
        step,
        dtype=jnp.float32,
    )

    # ── Schedule factory (Gaussian IQ pulse, 2D sweep over freq & dur) ─────
    def make_schedule(freq, dur):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=dur,
                amplitude=DRIVE_AMP_GHZ,
                frequency=freq,
                sigma=dur / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    # ── Simulate (2D sweep: freq × dur) ──────────────────────────────────────
    # tsave is a callable so each sweep point is measured at t=dur
    # (immediately after its pulse ends), avoiding spurious T1 decay.
    result = simulate_sweep(
        device,
        make_schedule,
        tsave=lambda dur, **_: jnp.array([0.0, dur]),
        freq=drive_freqs,
        dur=durations,
    )
    # result shape: (n_freq, n_dur, n_tsave) → take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.05, "Simulation should show spin-flip signal"

    # ── Build ds_raw ─────────────────────────────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "detuning": (np.asarray(drive_freqs) * 1e9, "qubit frequency", "Hz"),
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
            "frequency_span_in_mhz": FREQ_SPAN_MHZ,
            "frequency_step_in_mhz": FREQ_STEP_MHZ,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"FFT analysis should succeed: {fit_q1}"

    # Resonant frequency ≈ 10 GHz (wider tolerance for FFT estimates)
    f_res_ghz = fit_q1["optimal_frequency"] * 1e-9
    assert 9.0 < f_res_ghz < 11.0, f"Expected f_res ≈ 10 GHz, got {f_res_ghz:.3f}"

    # π-time in reasonable range
    t_pi = fit_q1["optimal_duration"]
    assert 30 < t_pi < 500, f"Expected t_π ∈ [30, 500] ns, got {t_pi:.0f}"

    # Decay rate should be present and finite
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma), f"Expected finite γ, got {gamma}"
