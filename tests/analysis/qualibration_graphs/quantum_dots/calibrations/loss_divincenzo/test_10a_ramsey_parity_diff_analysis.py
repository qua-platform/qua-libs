"""Analysis test for 10a_ramsey_parity_diff.

Uses virtual_qpu to simulate a 1-D Ramsey experiment (idle-time sweep at
a fixed drive-frequency detuning) for a Loss-DiVincenzo spin qubit.
The pi/2 pulse amplitude is pre-calibrated via a quick power-Rabi sweep
(``calibrated_pi_half_amp`` fixture in conftest).
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
MAX_TAU_NS = 800
N_TAU_POINTS = 200
DETUNING_MHZ = 1.0  # fixed drive-frequency detuning


@pytest.mark.analysis
def test_10a_ramsey_parity_diff_analysis(ld_device, calibrated_pi_half_amp, analysis_runner):
    """1-D Ramsey with pre-calibrated pi/2 pulse and damped-cosine fit."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_half_amp = calibrated_pi_half_amp

    # Drive frequency is offset from the qubit frequency
    detuning_ghz = DETUNING_MHZ * 1e-3
    drive_freq = qubit_freq_ghz + detuning_ghz

    # ── Sweep axis ────────────────────────────────────────────────────────
    tau_values = jnp.linspace(16, MAX_TAU_NS, N_TAU_POINTS, dtype=jnp.float32)

    # ── Ramsey schedule factory ──────────────────────────────────────────
    def make_ramsey_schedule(tau):
        sched = Schedule()
        # First pi/2 pulse
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_half_amp,
                frequency=drive_freq,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        # Idle wait
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
                frequency=drive_freq,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    # ── Simulate (1D sweep: tau only) ─────────────────────────────────────
    result = simulate_sweep(
        device,
        make_ramsey_schedule,
        tsave=lambda tau, **_: jnp.array([0.0, 2 * PI_HALF_DUR + tau]),
        tau=tau_values,
    )
    # result shape: (n_tau, n_tsave) -> take final time-point
    pdiff_q1 = result[..., -1]
    assert np.max(pdiff_q1) > 0.01, "Simulation should show some signal"

    # ── Build ds_raw (1D: tau only) ───────────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
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
            "frequency_detuning_in_mhz": DETUNING_MHZ,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # Ramsey frequency should be positive and in a physically reasonable
    # range.  The exact value depends on the actual qubit frequency in
    # the virtual QPU, which may differ slightly from the nominal value.
    ramsey_freq_mhz = fit_q1["ramsey_freq"] * 1e-6
    assert 0.1 < ramsey_freq_mhz < 10.0, (
        f"Ramsey freq should be in (0.1, 10) MHz, got {ramsey_freq_mhz:.3f} MHz"
    )

    # T2* should be finite and positive
    t2_star = fit_q1["t2_star"]
    assert np.isfinite(t2_star) and t2_star > 0, f"Expected finite T2* > 0, got {t2_star}"

    # Decay rate should be positive
    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma) and gamma > 0, f"Expected finite gamma > 0, got {gamma}"
