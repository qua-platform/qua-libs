"""Analysis test for 13_xy8.

Uses virtual_qpu with Lindblad master equation (solver="me") to simulate
the XY8 dynamical decoupling sequence for two qubits with different
decoherence parameters.  Both qubits use full physics simulation with
collapse operators — no synthetic data.

The XY8 sequence with CPMG timing is:

    π/2 – τ – X – 2τ – Y – 2τ – X – 2τ – Y – 2τ – Y – 2τ – X – 2τ – Y – 2τ – X – τ – π/2

Total idle time = 16τ.  The decay is fitted as P(τ) = offset + A·exp(−16τ/T₂_XY8).

Q1: t1=2000ns, t2=400ns — dephasing-dominated.
Q2: t1=5000ns, t2=600ns — different coherence regime.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule

from quantum_dots.device import LossDiVincenzoDevice

from .conftest import (
    ARTIFACTS_BASE,
    DEFAULT_DRIVE_AMP_GHZ,
    DEFAULT_PULSE_DURATION_NS,
    _VIRTUAL_QPU_AVAILABLE,
    build_joint_stream_analysis_ds,
    ld_params_with_decoherence,
    simulate_sweep,
)

NODE_NAME = "13_xy8"

PI_HALF_DUR = DEFAULT_PULSE_DURATION_NS  # ns
MAX_TAU_NS = 200
N_TAU_POINTS = 50
NOISE_STD = 0.02

# Input T2 (dephasing) for each device — fit should give T2_xy8 > T2
Q1_T2_INPUT_NS = 400.0
Q2_T2_INPUT_NS = 600.0

# XY8 pulse pattern: X Y X Y Y X Y X
_XY8_PHASES = [0.0, -np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, 0.0, -np.pi / 2, 0.0]


def _make_xy8_device(t1: float, t2: float) -> LossDiVincenzoDevice:
    """Create a 2-qubit device with T1/T2 decoherence for Lindblad simulation."""
    params = ld_params_with_decoherence([t1, t1], [t2, t2])
    return LossDiVincenzoDevice(params=params)


def _calibrate_pi_half_amp(device: LossDiVincenzoDevice) -> float:
    """Quick power-Rabi calibration to find the pi/2 amplitude."""
    qubit_freq_ghz = device.params.qubit_freqs[0]
    amp_prefactors = jnp.linspace(0.1, 3.0, 200, dtype=jnp.float32)

    def make_schedule(amp):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=DEFAULT_DRIVE_AMP_GHZ * amp,
                frequency=qubit_freq_ghz,
                sigma=PI_HALF_DUR / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    result = simulate_sweep(
        device,
        make_schedule,
        tsave=jnp.array([0.0, PI_HALF_DUR], dtype=jnp.float32),
        noise_std=0.0,
        amp=amp_prefactors,
    )
    parity = np.asarray(result[..., -1])
    pi_idx = int(np.argmax(parity))
    pi_amp = float(DEFAULT_DRIVE_AMP_GHZ * amp_prefactors[pi_idx])
    return pi_amp / 2.0


def _build_xy8_schedule(tau, pi_half_amp, pi_amp, qubit_freq_ghz):
    """Build (but do not resolve) an XY8 schedule with CPMG timing for a given τ."""
    sched = Schedule()

    # Opening pi/2
    sched.play(
        GaussianIQPulse(
            duration=PI_HALF_DUR,
            amplitude=pi_half_amp,
            frequency=qubit_freq_ghz,
            sigma=PI_HALF_DUR / 5,
        ),
        channel="drive_q0",
    )

    # First half-interval (τ)
    sched.play(
        SquarePulse(duration=tau, amplitude=0.0, frequency=0.0),
        channel="drive_q0",
    )

    for i, phase in enumerate(_XY8_PHASES):
        sched.play(
            GaussianIQPulse(
                duration=PI_HALF_DUR,
                amplitude=pi_amp,
                frequency=qubit_freq_ghz,
                sigma=PI_HALF_DUR / 5,
                phase=phase,
            ),
            channel="drive_q0",
        )
        idle_duration = tau if i == 7 else 2 * tau
        sched.play(
            SquarePulse(duration=idle_duration, amplitude=0.0, frequency=0.0),
            channel="drive_q0",
        )

    # Closing pi/2
    sched.play(
        GaussianIQPulse(
            duration=PI_HALF_DUR,
            amplitude=pi_half_amp,
            frequency=qubit_freq_ghz,
            sigma=PI_HALF_DUR / 5,
        ),
        channel="drive_q0",
    )

    return sched


def _simulate_xy8_qubit(
    device: LossDiVincenzoDevice,
    pi_half_amp: float,
    tau_values: jnp.ndarray,
) -> np.ndarray:
    """Run XY8 Lindblad simulation for one qubit configuration."""
    qubit_freq_ghz = device.params.qubit_freqs[0]
    pi_amp = 2.0 * pi_half_amp

    def make_xy8_schedule(tau):
        return _build_xy8_schedule(tau, pi_half_amp, pi_amp, qubit_freq_ghz).resolve()

    total_pulse_time = 10 * PI_HALF_DUR  # 8 pi pulses + 2 pi/2 pulses

    result = simulate_sweep(
        device,
        make_xy8_schedule,
        tsave=lambda tau, **_: jnp.array(
            [0.0, total_pulse_time + 16 * tau], dtype=jnp.float32
        ),
        solver="me",
        noise_std=NOISE_STD,
        tau=tau_values,
    )
    return result[..., -1]


@pytest.mark.analysis
def test_13_xy8_analysis(analysis_runner):
    """XY8 dynamical decoupling with Lindblad simulation for two qubits."""
    if not _VIRTUAL_QPU_AVAILABLE:
        pytest.skip("virtual_qpu (dynamiqs) not installed — skipping Lindblad XY8 test")

    tau_values = jnp.linspace(16, MAX_TAU_NS, N_TAU_POINTS, dtype=jnp.float32)

    # ── Q1: t1=2000ns, t2=400ns (dephasing-dominated) ────────────────
    device_q1 = _make_xy8_device(t1=2000.0, t2=Q1_T2_INPUT_NS)
    pi_half_amp_q1 = _calibrate_pi_half_amp(device_q1)
    pdiff_q1 = _simulate_xy8_qubit(device_q1, pi_half_amp_q1, tau_values)
    assert np.max(pdiff_q1) > 0.01, "Q1 simulation should show some signal"

    # ── Q2: t1=5000ns, t2=600ns (longer coherence) ───────────────────
    device_q2 = _make_xy8_device(t1=5000.0, t2=Q2_T2_INPUT_NS)
    pi_half_amp_q2 = _calibrate_pi_half_amp(device_q2)
    pdiff_q2 = _simulate_xy8_qubit(device_q2, pi_half_amp_q2, tau_values)
    assert np.max(pdiff_q2) > 0.01, "Q2 simulation should show some signal"

    # ── Save schedule plot ────────────────────────────────────────────
    qubit_freq_ghz = device_q1.params.qubit_freqs[0]
    first_tau = float(tau_values[0])
    sched_first = _build_xy8_schedule(
        first_tau, pi_half_amp_q1, 2.0 * pi_half_amp_q1, qubit_freq_ghz
    )
    ax = sched_first.plot()
    ax.set_title(f"XY8 schedule (τ = {first_tau:.0f} ns)")
    fig_sched = ax.get_figure()
    fig_sched.tight_layout()
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fig_sched.savefig(artifacts_dir / "schedule.png", dpi=200)
    plt.close(fig_sched)

    # Joint-stream format: p0_p0 + E_p2_given_p1_0 (required by fit_raw_data)
    tau_np = np.asarray(tau_values, dtype=np.float64)
    ds_raw = build_joint_stream_analysis_ds(
        coords={"tau": (tau_np, "half inter-pulse spacing", "ns")},
        signal_per_qubit={"q1": pdiff_q1, "q2": pdiff_q2},
    )

    # ── Run analysis ──────────────────────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "tau_min": 16,
            "tau_max": int(MAX_TAU_NS),
            "tau_step": 4,
        },
        analyse_qubits=["q1", "q2"],
    )

    # ── Assertions ────────────────────────────────────────────────────
    assert "fit_results" in node.results

    # Q1: dephasing-dominated
    fit_q1 = node.results["fit_results"]["q1"]
    assert fit_q1["success"], f"q1 analysis should succeed: {fit_q1}"
    assert fit_q1["T2_xy8"] > 0, f"q1 T2_xy8 should be positive, got {fit_q1['T2_xy8']}"
    assert np.isfinite(
        fit_q1["T2_xy8"]
    ), f"q1 T2_xy8 should be finite, got {fit_q1['T2_xy8']}"
    assert (
        fit_q1["decay_rate"] > 0
    ), f"q1 decay_rate should be positive, got {fit_q1['decay_rate']}"
    assert (
        fit_q1["T2_xy8"] > Q1_T2_INPUT_NS
    ), f"q1: T2_xy8 should exceed input T2={Q1_T2_INPUT_NS} ns, got {fit_q1['T2_xy8']:.1f} ns"

    # Q2: longer input T2
    fit_q2 = node.results["fit_results"]["q2"]
    assert fit_q2["success"], f"q2 analysis should succeed: {fit_q2}"
    assert fit_q2["T2_xy8"] > 0, f"q2 T2_xy8 should be positive, got {fit_q2['T2_xy8']}"
    assert np.isfinite(
        fit_q2["T2_xy8"]
    ), f"q2 T2_xy8 should be finite, got {fit_q2['T2_xy8']}"
    assert (
        fit_q2["decay_rate"] > 0
    ), f"q2 decay_rate should be positive, got {fit_q2['decay_rate']}"
    assert (
        fit_q2["T2_xy8"] > Q2_T2_INPUT_NS
    ), f"q2: T2_xy8 should exceed input T2={Q2_T2_INPUT_NS} ns, got {fit_q2['T2_xy8']:.1f} ns"

    # Distinct effective coherence times (different Lindblad inputs)
    assert not np.isclose(
        fit_q1["T2_xy8"],
        fit_q2["T2_xy8"],
        rtol=0.02,
    ), f"q1 and q2 fits should differ, got T2_xy8 {fit_q1['T2_xy8']:.1f} vs {fit_q2['T2_xy8']:.1f} ns"
