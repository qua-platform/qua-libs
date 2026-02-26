"""Analysis test for 08a_power_rabi.

Uses virtual_qpu to simulate a Lindblad power-Rabi oscillation for a
Loss-DiVincenzo spin qubit with a Gaussian IQ pulse, builds a synthetic
1D ``ds_raw`` (amplitude prefactor only), and runs the FFT-seeded
damped-sinusoid analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse
from virtual_qpu.schedule import Schedule

from .conftest import ANALYSE_QUBITS, build_parity_ds_raw, simulate_sweep

NODE_NAME = "08a_power_rabi"

# ── Node-specific simulation parameters ─────────────────────────────────────
DRIVE_AMP_GHZ = 0.008  # base amplitude (GHz)
PULSE_DURATION_NS = 100  # fixed pulse duration for power sweep
N_AMP_POINTS = 200


@pytest.mark.analysis
def test_08a_power_rabi_analysis(ld_device, analysis_runner):
    """Lindblad power-Rabi (Gaussian pulse) with damped-sinusoid analysis."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]

    # ── Sweep axis ───────────────────────────────────────────────────────────
    amp_prefactors = jnp.linspace(0.001, 1.99, N_AMP_POINTS, dtype=jnp.float32)

    # ── Schedule factory (Gaussian IQ pulse, on-resonance, amplitude sweep) ──
    def make_schedule(amp):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=PULSE_DURATION_NS,
                amplitude=DRIVE_AMP_GHZ * amp,
                frequency=qubit_freq_ghz,
                sigma=PULSE_DURATION_NS / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    # ── Simulate (1D sweep: amplitude prefactor only) ────────────────────────
    result = simulate_sweep(
        device,
        make_schedule,
        tsave=jnp.array([0.0, PULSE_DURATION_NS], dtype=jnp.float32),
        amp=amp_prefactors,
    )
    # result shape: (n_amp, n_tsave) → take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.05, "Simulation should show spin-flip signal"

    # ── Build ds_raw (1D: amp_prefactor only) ────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "amp_prefactor": (np.asarray(amp_prefactors), "pulse amplitude prefactor", ""),
        },
        pdiff_per_qubit={q: pdiff_q1 for q in ANALYSE_QUBITS},
    )

    # ── Run analysis ─────────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_amp_factor": 0.001,
            "max_amp_factor": 1.99,
            "amp_factor_step": 1.99 / N_AMP_POINTS,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # Optimal amplitude prefactor in reasonable range
    a_pi = fit_q1["opt_amp"]
    assert 0.05 < a_pi < 1.95, f"Expected a_pi in [0.05, 1.95], got {a_pi:.4f}"

    # Rabi frequency should be finite and positive
    omega = fit_q1["rabi_frequency"]
    assert np.isfinite(omega) and omega > 0, f"Expected finite Omega > 0, got {omega}"

    # Decay rate should be finite
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma), f"Expected finite gamma, got {gamma}"

    # Damped sinusoid fit should have been used (not just FFT fallback)
    sinusoid = fit_q1.get("_sinusoid_fit")
    assert sinusoid is not None, "Damped sinusoid fit should succeed on clean simulation data"
    assert sinusoid["frequency"] > 0, "Fitted frequency should be positive"
    assert sinusoid["gamma"] >= 0, "Fitted decay rate should be non-negative"
    # Consistency: sinusoid frequency should agree with reported Rabi values
    omega_from_sin = 2.0 * np.pi * sinusoid["frequency"]
    assert (
        abs(omega_from_sin - omega) / omega < 0.01
    ), f"Sinusoid freq and reported Omega should agree: {omega_from_sin:.5f} vs {omega:.5f}"
