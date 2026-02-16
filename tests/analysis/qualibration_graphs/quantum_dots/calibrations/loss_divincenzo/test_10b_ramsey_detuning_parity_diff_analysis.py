"""Analysis test for 10b_ramsey_detuning_parity_diff (detuning sweep).

Uses virtual_qpu to simulate a Ramsey experiment at fixed idle time τ,
sweeping the drive-frequency detuning across a span around the qubit
resonance.

The parity-difference signal is a cosine in detuning whose period is
set by the known τ.  A linear cosine decomposition extracts the
resonance detuning δ₀ and contrast.

The π/2 pulse amplitude is pre-calibrated via the
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

NODE_NAME = "10b_ramsey_detuning_parity_dif"

# ── Simulation parameters ────────────────────────────────────────────────────
PI_HALF_DUR = DEFAULT_PULSE_DURATION_NS  # ns
IDLE_TIME_NS = 200  # fixed idle time
DETUNING_SPAN_MHZ = 10.0  # sweep ±5 MHz around qubit frequency
DETUNING_STEP_MHZ = 0.1
N_DETUNING = int(DETUNING_SPAN_MHZ / DETUNING_STEP_MHZ)


@pytest.mark.analysis
def test_10b_ramsey_detuning_parity_diff_analysis(
    ld_device, calibrated_pi_half_amp, analysis_runner
):
    """Detuning-sweep Ramsey with linear cosine decomposition."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_half_amp = calibrated_pi_half_amp

    # ── Detuning sweep: drive frequencies across ±span/2 around qubit ────
    detuning_hz = np.arange(
        -DETUNING_SPAN_MHZ / 2 * 1e6,
        DETUNING_SPAN_MHZ / 2 * 1e6,
        DETUNING_STEP_MHZ * 1e6,
    )
    drive_freqs_ghz = qubit_freq_ghz + detuning_hz * 1e-9

    # ── Ramsey schedule factory (freq = drive frequency in GHz) ──────────
    idle_time = jnp.float32(IDLE_TIME_NS)

    def make_ramsey_schedule(freq):
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
                duration=idle_time,
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

    # ── Simulate (1D sweep over drive frequencies) ───────────────────────
    total_dur = 2 * PI_HALF_DUR + IDLE_TIME_NS
    result = simulate_sweep(
        device,
        make_ramsey_schedule,
        tsave=jnp.array([0.0, total_dur], dtype=jnp.float32),
        freq=jnp.array(drive_freqs_ghz, dtype=jnp.float32),
    )
    # result shape: (n_detuning, n_tsave) → take final time-point
    pdiff_1d = result[..., -1]  # (n_detuning,)
    assert pdiff_1d.shape == (len(detuning_hz),)
    assert np.max(pdiff_1d) > 0.01, "Simulation should show some signal"

    # ── Build ds_raw (1D: detuning) ──────────────────────────────────────
    ds_raw = build_parity_ds_raw(
        coords={
            "detuning": (detuning_hz, "frequency detuning", "Hz"),
        },
        pdiff_per_qubit={q: pdiff_1d for q in ANALYSE_QUBITS},
    )

    # ── Run analysis ─────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "detuning_span_in_mhz": DETUNING_SPAN_MHZ,
            "detuning_step_in_mhz": DETUNING_STEP_MHZ,
            "idle_time_ns": IDLE_TIME_NS,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["Q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    # freq_offset should be within the detuning span
    freq_offset_mhz = fit_q1["freq_offset"] * 1e-6
    half_span = DETUNING_SPAN_MHZ / 2
    assert abs(freq_offset_mhz) < half_span, (
        f"freq_offset should be within ±{half_span} MHz, got {freq_offset_mhz:.4f} MHz"
    )

    # Contrast should be positive
    contrast = fit_q1["contrast"]
    assert contrast > 0, f"Expected positive contrast, got {contrast}"
