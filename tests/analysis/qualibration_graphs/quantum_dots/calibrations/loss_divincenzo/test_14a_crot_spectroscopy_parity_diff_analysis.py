"""Analysis test for 14a_crot_spectroscopy_parity_diff.

Uses virtual_qpu to simulate CROT spectroscopy on a 2-qubit
Loss-DiVincenzo spin chain.  The target qubit's resonance frequency
depends on the control qubit's spin state via the ZZ exchange coupling:

    f_target(control=|↑⟩) = f_0 + J/2
    f_target(control=|↓⟩) = f_0 − J/2

The experiment sweeps ESR drive frequency × exchange voltage for two
control-qubit preparations (no x180 → control in |↑⟩ ground;
with x180 → control in |↓⟩ excited), producing a pair of 2-D parity-
diff maps whose peak splitting gives J(V).

The analysis fits Lorentzians to each frequency slice and extracts
the exchange coupling J, crot_frequency_down, and crot_frequency_up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import jax.numpy as jnp

from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule

from .conftest import (
    ARTIFACTS_BASE,
    DEFAULT_PULSE_DURATION_NS,
    QUBIT_PAIR_NAMES,
    build_parity_ds_raw,
    simulate_sweep,
)

NODE_NAME = "14a_crot_spectroscopy_parity_diff"

# ── Simulation parameters ────────────────────────────────────────────────────
CROT_PULSE_DURATION = 200.0
ESR_AMP_GHZ = 0.003
PI_DUR = DEFAULT_PULSE_DURATION_NS
NOISE_STD = 0.02

N_EXCHANGE = 35
N_ESR_FREQ = 51
EXCHANGE_V_MIN = -0.15
EXCHANGE_V_MAX = 0.02
ESR_FREQ_HALF_SPAN_HZ = 15e6


@pytest.mark.analysis
def test_14a_crot_spectroscopy_analysis(
    ld_device,
    calibrated_pi_half_amp,
    calibrated_control_pi_amp,
    analysis_runner,
):
    """CROT spectroscopy: extract J from two 2-D parity-diff maps."""
    device = ld_device
    target_freq_ghz = device.params.qubit_freqs[0]
    control_freq_ghz = device.params.qubit_freqs[1]
    control_pi_amp = calibrated_control_pi_amp

    # ── Sweep axes ────────────────────────────────────────────────────────
    target_freq_hz = target_freq_ghz * 1e9
    exchange_voltages = jnp.linspace(
        EXCHANGE_V_MIN,
        EXCHANGE_V_MAX,
        N_EXCHANGE,
        dtype=jnp.float32,
    )
    esr_freq_abs_hz = jnp.linspace(
        target_freq_hz - ESR_FREQ_HALF_SPAN_HZ,
        target_freq_hz + ESR_FREQ_HALF_SPAN_HZ,
        N_ESR_FREQ,
        dtype=jnp.float32,
    )

    # ── Schedule factories ────────────────────────────────────────────────

    def _voltage_to_j(v):
        """Exponential exchange model: J(V) = J_0 * exp((V - V_ref) / λ)."""
        return EXCHANGE_J0_GHZ * jnp.exp((v - EXCHANGE_V_REF) / EXCHANGE_LEVER_ARM)

    def make_crot_schedule_no_x180(exchange_voltage, esr_freq):
        """ESR spectroscopy drive + exchange pulse, control qubit untouched."""
        j_ghz = _voltage_to_j(exchange_voltage)
        sched = Schedule()
        sched.play(
            SquarePulse(
                duration=CROT_PULSE_DURATION,
                amplitude=j_ghz,
                frequency=0.0,
            ),
            channel="exchange_0_1",
        )
        sched.play(
            GaussianIQPulse(
                duration=CROT_PULSE_DURATION,
                amplitude=ESR_AMP_GHZ,
                frequency=esr_freq * 1e-9,
                sigma=CROT_PULSE_DURATION / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    def make_crot_schedule_with_x180(exchange_voltage, esr_freq):
        """Pi pulse on control qubit, then ESR + exchange."""
        j_ghz = _voltage_to_j(exchange_voltage)
        sched = Schedule()
        pi_ref = sched.play(
            GaussianIQPulse(
                duration=PI_DUR,
                amplitude=control_pi_amp,
                frequency=control_freq_ghz,
                sigma=PI_DUR / 5,
            ),
            channel="drive_q1",
        )
        sched.play(
            SquarePulse(
                duration=CROT_PULSE_DURATION,
                amplitude=j_ghz,
                frequency=0.0,
            ),
            channel="exchange_0_1",
            after=[pi_ref],
        )
        sched.play(
            GaussianIQPulse(
                duration=CROT_PULSE_DURATION,
                amplitude=ESR_AMP_GHZ,
                frequency=esr_freq * 1e-9,
                sigma=CROT_PULSE_DURATION / 5,
            ),
            channel="drive_q0",
            after=[pi_ref],
        )
        return sched.resolve()

    # ── Save a schedule plot for the first sweep point ────────────────────
    first_v = float(exchange_voltages[0])
    first_f = float(esr_freq_abs_hz[N_ESR_FREQ // 2])
    first_j = float(_voltage_to_j(first_v))
    sched_example = Schedule()
    sched_example.play(
        SquarePulse(duration=CROT_PULSE_DURATION, amplitude=first_j, frequency=0.0),
        channel="exchange_0_1",
    )
    sched_example.play(
        GaussianIQPulse(
            duration=CROT_PULSE_DURATION,
            amplitude=ESR_AMP_GHZ,
            frequency=first_f * 1e-9,
            sigma=CROT_PULSE_DURATION / 5,
        ),
        channel="drive_q0",
    )
    ax = sched_example.plot()
    ax.set_title("CROT spectroscopy schedule (no x180)")
    fig_sched = ax.get_figure()
    fig_sched.tight_layout()
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fig_sched.savefig(artifacts_dir / "schedule.png", dpi=200)
    plt.close(fig_sched)

    # ── Simulate: control_x180 = False (control in ground |↑⟩) ───────────
    result_no_x180 = simulate_sweep(
        device,
        make_crot_schedule_no_x180,
        tsave=jnp.array([0.0, CROT_PULSE_DURATION], dtype=jnp.float32),
        observable_qubit=0,
        noise_std=NOISE_STD,
        seed=42,
        exchange_voltage=exchange_voltages,
        esr_freq=esr_freq_abs_hz,
    )
    pdiff_no_x180 = result_no_x180[..., -1]  # (n_exchange, n_esr_freq)

    # ── Simulate: control_x180 = True (control flipped to |↓⟩) ───────────
    result_with_x180 = simulate_sweep(
        device,
        make_crot_schedule_with_x180,
        tsave=jnp.array(
            [0.0, PI_DUR + CROT_PULSE_DURATION],
            dtype=jnp.float32,
        ),
        observable_qubit=0,
        noise_std=NOISE_STD,
        seed=99,
        exchange_voltage=exchange_voltages,
        esr_freq=esr_freq_abs_hz,
    )
    pdiff_with_x180 = result_with_x180[..., -1]  # (n_exchange, n_esr_freq)

    assert pdiff_no_x180.shape == (N_EXCHANGE, N_ESR_FREQ)
    assert pdiff_with_x180.shape == (N_EXCHANGE, N_ESR_FREQ)

    # ── Assemble 3-D dataset: (control_x180, exchange, esr_frequency) ─────
    pdiff_3d = np.stack(
        [np.asarray(pdiff_no_x180), np.asarray(pdiff_with_x180)],
        axis=0,
    )

    esr_freq_np = np.asarray(esr_freq_abs_hz, dtype=np.float64)
    exchange_np = np.asarray(exchange_voltages, dtype=np.float64)

    ds_raw = build_parity_ds_raw(
        coords={
            "control_x180": (
                np.array([False, True]),
                "x180 on control qubit",
                "bool",
            ),
            "exchange": (exchange_np, "voltage", "V"),
            "esr_frequency": (esr_freq_np, "frequency", "Hz"),
        },
        pdiff_per_qubit={"q1_q2": pdiff_3d},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    # ── Run the node's analyse_data → plot_data → update_state chain ──────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "exchange_min": float(EXCHANGE_V_MIN),
            "exchange_max": float(EXCHANGE_V_MAX),
            "exchange_points": N_EXCHANGE,
            "esr_frequency_min": float(-ESR_FREQ_HALF_SPAN_HZ),
            "esr_frequency_max": float(ESR_FREQ_HALF_SPAN_HZ),
            "esr_frequency_points": N_ESR_FREQ,
            "duration": int(CROT_PULSE_DURATION),
            "hold_duration": 100,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    # ── Assertions ────────────────────────────────────────────────────────
    assert "fit_results" in node.results, "analyse_data should populate fit_results"

    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"CROT fit should succeed: {fit}"
    assert fit["exchange_coupling_J"] > 0, f"J should be positive, got {fit['exchange_coupling_J']}"
    assert np.isfinite(fit["exchange_coupling_J"]), f"J should be finite, got {fit['exchange_coupling_J']}"
    assert np.isfinite(fit["crot_frequency_down"]), f"f_down should be finite, got {fit['crot_frequency_down']}"
    assert np.isfinite(fit["crot_frequency_up"]), f"f_up should be finite, got {fit['crot_frequency_up']}"

    j_from_splitting = abs(fit["crot_frequency_up"] - fit["crot_frequency_down"])
    assert (
        abs(j_from_splitting - fit["exchange_coupling_J"]) < 0.3 * fit["exchange_coupling_J"]
    ), f"|f_up - f_down| = {j_from_splitting:.0f} Hz should ≈ J = {fit['exchange_coupling_J']:.0f} Hz"
