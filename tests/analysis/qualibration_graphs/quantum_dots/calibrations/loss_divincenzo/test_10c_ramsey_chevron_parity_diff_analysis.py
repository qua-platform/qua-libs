"""Analysis test for 10c_ramsey_chevron_parity_diff.

Uses virtual_qpu to simulate a Ramsey chevron (2D: detuning x idle time)
for a Loss-DiVincenzo spin qubit.  The pi/2 pulse amplitude is pre-
calibrated via a quick power-Rabi sweep (``calibrated_pi_half_amp``
fixture in conftest).
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

NODE_NAME = "10c_ramsey_chevron_parity_diff"

# ── Simulation parameters ────────────────────────────────────────────────────
PI_HALF_DUR = DEFAULT_PULSE_DURATION_NS  # ns (same duration used for calibration)
MAX_TAU_NS = 400
N_TAU_POINTS = 100
DETUNING_SPAN_MHZ = 10.0
DETUNING_STEP_MHZ = 0.2


@pytest.mark.analysis
def test_10c_ramsey_chevron_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """Ramsey chevron with pre-calibrated pi/2 pulse and per-slice FFT analysis."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_half_amp = calibrated_pi_half_amp

    # ── Sweep axes ───────────────────────────────────────────────────────
    tau_values = jnp.linspace(16, MAX_TAU_NS, N_TAU_POINTS, dtype=jnp.float32)

    span_ghz = DETUNING_SPAN_MHZ * 1e-3
    step_ghz = DETUNING_STEP_MHZ * 1e-3
    drive_freqs = jnp.arange(
        qubit_freq_ghz - span_ghz / 2,
        qubit_freq_ghz + span_ghz / 2,
        step_ghz,
        dtype=jnp.float32,
    )

    # ── Ramsey schedule factory ──────────────────────────────────────────
    def make_ramsey_schedule(freq, tau):
        sched = Schedule()
        # First pi/2 pulse
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=freq,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        # Idle wait (zero-amplitude pulse to reserve time)
        sched.play(
            SquarePulse(
                duration=tau,
                amplitude=0.0,
                frequency=0.0,
            ),
            channel="drive_q0",
        )
        # Second pi/2 pulse
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

    # ── Simulate (2D sweep: freq x tau) ──────────────────────────────────
    result = simulate_sweep(
        device,
        make_ramsey_schedule,
        tsave=lambda tau, **_: jnp.array([0.0, 2 * PI_HALF_DUR + tau]),
        freq=drive_freqs,
        tau=tau_values,
    )
    # result shape: (n_freq, n_tau, n_tsave) -> take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.01, "Simulation should show some signal"

    # ── Build ds_raw (2D: detuning x tau) ────────────────────────────────
    # Convert drive_freqs from GHz to Hz detuning for the dataset
    detuning_hz = (np.asarray(drive_freqs) - qubit_freq_ghz) * 1e9

    ds_raw = build_parity_ds_raw(
        coords={
            "detuning": (detuning_hz, "frequency detuning", "Hz"),
            "tau": (np.asarray(tau_values), "idle time", "ns"),
        },
        pdiff_per_qubit={q: pdiff_q1 for q in ANALYSE_QUBITS},
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
            "detuning_span_in_mhz": DETUNING_SPAN_MHZ,
            "detuning_step_in_mhz": DETUNING_STEP_MHZ,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # Frequency offset should be near 0 (we drove on resonance)
    freq_offset_mhz = fit_q1["freq_offset"] * 1e-6
    assert (
        abs(freq_offset_mhz) < DETUNING_SPAN_MHZ / 2
    ), f"freq_offset should be within sweep range, got {freq_offset_mhz:.3f} MHz"

    # T2* should be finite and positive
    t2_star = fit_q1["t2_star"]
    assert np.isfinite(t2_star) and t2_star > 0, f"Expected finite T2* > 0, got {t2_star}"

    # At least one decay component must be positive
    gamma = fit_q1["decay_rate"]
    sigma_g = fit_q1["gauss_decay_rate"]
    assert np.isfinite(gamma) and gamma >= 0, f"Expected finite gamma >= 0, got {gamma}"
    assert np.isfinite(sigma_g) and sigma_g >= 0, f"Expected finite sigma_g >= 0, got {sigma_g}"
    assert gamma > 0 or sigma_g > 0, "At least one decay component must be positive"
