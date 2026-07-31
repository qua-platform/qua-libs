"""Analysis test for 09b_power_rabi_error_amplification.

Uses virtual_qpu to simulate a 2D error-amplified power-Rabi sweep
(n_pulses × amplitude prefactor) for a Loss-DiVincenzo spin qubit with
Gaussian IQ pulses, builds a synthetic ``ds_raw``, and runs the
mean-signal resonance-finding analysis pipeline (Ramsey-chevron style).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse
from virtual_qpu.schedule import Schedule

from .conftest import ANALYSE_QUBITS, build_joint_stream_analysis_ds, simulate_sweep

NODE_NAME = "09b_power_rabi_error_amplification"

# ── Node-specific simulation parameters ─────────────────────────────────────
DRIVE_AMP_GHZ = 0.008
PULSE_DURATION_NS = 100
N_AMP_POINTS = 200
MIN_AMP = 0.45
MAX_AMP = 0.65
MAX_N_PULSES = 42  # arange(2, 42, 2) → [2, 4, ..., 40]


@pytest.mark.analysis
def test_09b_power_rabi_error_amplification_analysis(ld_device, analysis_runner):
    """Lindblad error-amplified power-Rabi with mean-signal resonance analysis."""
    device = ld_device
    qubit_freq_ghz = device.params.qubit_freqs[0]

    # ── Sweep axes ───────────────────────────────────────────────────────────
    amp_prefactors = jnp.linspace(MIN_AMP, MAX_AMP, N_AMP_POINTS, dtype=jnp.float32)
    n_pulses_values = np.arange(2, MAX_N_PULSES, 2)

    # ── Simulate (2D sweep: n_pulses × amp) ──────────────────────────────────
    # n_pulses controls schedule *structure* (number of pulse ops) so it can't
    # be JAX-traced via vmap.  Loop over n_pulses in Python; vmap over amp.
    pdiff_slices = []
    for n_p in n_pulses_values:
        n_p_int = int(n_p)

        def _make_schedule(amp, _n=n_p_int):
            sched = Schedule()
            for _ in range(_n):
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

        slice_result = simulate_sweep(
            device,
            _make_schedule,
            tsave=jnp.array([0.0, n_p_int * PULSE_DURATION_NS], dtype=jnp.float32),
            amp=amp_prefactors,
        )
        pdiff_slices.append(np.asarray(slice_result[..., -1]))

    pdiff_2d = np.stack(pdiff_slices, axis=0)
    assert np.max(pdiff_2d) > 0.05, "Simulation should show spin-flip signal"

    # ── Build ds_raw (2D: n_pulses × amp_prefactor, joint-stream format) ─────
    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "n_pulses": (
                n_pulses_values.astype(float),
                "number of pi pulses",
                "",
            ),
            "amp_prefactor": (
                np.asarray(amp_prefactors),
                "pulse amplitude prefactor",
                "",
            ),
        },
        signal_per_qubit={q: pdiff_2d for q in ANALYSE_QUBITS},
    )

    # ── Run analysis ─────────────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_amp_factor": MIN_AMP,
            "max_amp_factor": MAX_AMP,
            "amp_factor_step": (MAX_AMP - MIN_AMP) / N_AMP_POINTS,
            "max_n_pulses": MAX_N_PULSES,
        },
    )

    # ── Assertions ───────────────────────────────────────────────────────────
    assert "fit_results" in node.results
    fit_q1 = node.results["fit_results"]["q1"]
    assert fit_q1["success"], f"Analysis should succeed: {fit_q1}"

    a_pi = fit_q1["opt_amp"]
    assert (
        MIN_AMP < a_pi < MAX_AMP
    ), f"Expected a_pi in [{MIN_AMP}, {MAX_AMP}], got {a_pi:.4f}"

    omega = fit_q1["rabi_frequency"]
    assert np.isfinite(omega) and omega > 0, f"Expected finite Omega > 0, got {omega}"

    gamma = fit_q1["decay_rate"]
    assert np.isfinite(gamma), f"Expected finite gamma, got {gamma}"
