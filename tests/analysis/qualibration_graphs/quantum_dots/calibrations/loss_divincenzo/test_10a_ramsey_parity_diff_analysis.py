"""Analysis test for 10a_ramsey_parity_diff (±δ triangulation).

Uses virtual_qpu to simulate a Ramsey experiment at two symmetric
detunings ±δ from the qubit frequency.  Fitting a damped cosine to
each trace and triangulating gives the residual frequency offset
Δ = (f₋ − f₊) / 2.

The pi/2 pulse amplitude is pre-calibrated via the
``calibrated_pi_half_amp`` fixture in conftest.
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

NODE_NAME = "10a_ramsey_parity_diff"

# ── Simulation parameters ────────────────────────────────────────────────────
PI_HALF_DUR = DEFAULT_PULSE_DURATION_NS  # ns
MAX_TAU_NS = 1500
N_TAU_POINTS = 300
DETUNING_MHZ = 3.0  # applied ±δ detuning


@pytest.mark.analysis
def test_10a_ramsey_parity_diff_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """±δ Ramsey with pre-calibrated pi/2 pulse and triangulated frequency."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_half_amp = calibrated_pi_half_amp

    # ── Two symmetric drive frequencies: qubit_freq ± δ ──────────────────
    detuning_ghz = DETUNING_MHZ * 1e-3
    drive_freqs = [
        qubit_freq_ghz + detuning_ghz,   # +δ
        qubit_freq_ghz - detuning_ghz,   # -δ
    ]

    # ── Sweep axis ────────────────────────────────────────────────────────
    tau_values = jnp.linspace(16, MAX_TAU_NS, N_TAU_POINTS, dtype=jnp.float32)

    # ── Ramsey schedule factory (freq is the drive frequency) ────────────
    def make_ramsey_schedule(freq, tau):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=freq,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        sched.play(
            SquarePulse(
                duration=tau,
                amplitude=0.0,
                frequency=0.0,
            ),
            channel="drive_q0",
        )
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=freq,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    # ── Simulate (2D sweep: freq × tau) ──────────────────────────────────
    result = simulate_sweep(
        device,
        make_ramsey_schedule,
        tsave=lambda tau, **_: jnp.array([0.0, 2 * PI_HALF_DUR + tau]),
        freq=jnp.array(drive_freqs, dtype=jnp.float32),
        tau=tau_values,
    )
    # result shape: (n_freq, n_tau, n_tsave) → take final time-point
    pdiff_2d = result[..., -1]  # (2, n_tau)
    assert pdiff_2d.shape == (2, len(tau_values))
    assert np.max(pdiff_2d) > 0.01, "Simulation should show some signal"

    # ── Build ds_raw (2D: detuning × tau) ─────────────────────────────────
    detuning_hz = np.array([DETUNING_MHZ * 1e6, -DETUNING_MHZ * 1e6])

    ds_raw = build_parity_ds_raw(
        coords={
            "detuning": (detuning_hz, "frequency detuning", "Hz"),
            "tau": (np.asarray(tau_values), "idle time", "ns"),
        },
        pdiff_per_qubit={q: pdiff_2d for q in ANALYSE_QUBITS},
    )

    # ── Run analysis ─────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": MAX_TAU_NS,
            "wait_time_num_points": N_TAU_POINTS,
            "frequency_detuning_in_mhz": DETUNING_MHZ,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # Both fitted Ramsey frequencies should be positive and reasonable
    f_plus_mhz = fit_q1["freq_plus"] * 1e-6
    f_minus_mhz = fit_q1["freq_minus"] * 1e-6
    assert 0.1 < f_plus_mhz < 10.0, f"f₊ should be in (0.1, 10) MHz, got {f_plus_mhz:.3f}"
    assert 0.1 < f_minus_mhz < 10.0, f"f₋ should be in (0.1, 10) MHz, got {f_minus_mhz:.3f}"

    # Triangulated freq_offset should be small — the qubit is near the
    # nominal frequency (virtual QPU may have a small systematic offset)
    freq_offset_mhz = fit_q1["freq_offset"] * 1e-6
    assert abs(freq_offset_mhz) < DETUNING_MHZ, (
        f"Triangulated offset should be smaller than δ, got {freq_offset_mhz:.3f} MHz"
    )

    # T2* should be finite and positive
    t2_star = fit_q1["t2_star"]
    assert np.isfinite(t2_star) and t2_star > 0, f"Expected finite T2* > 0, got {t2_star}"

    # Decay rate should be positive
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma) and gamma > 0, f"Expected finite gamma > 0, got {gamma}"
