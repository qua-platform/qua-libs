"""Analysis test for 18a_swap_oscillations.

Uses virtual_qpu (LossDiVincenzoDevice + dynamiqs) to simulate swap
oscillations as a function of exchange amplitude and duration.

Starting from |10⟩ (target qubit excited via perfect X180 unitary), the
exchange interaction causes Heisenberg flip-flop between the two qubits.
The single-qubit |1⟩⟨1| projectors on each qubit give complementary
oscillation patterns whose frequency grows exponentially with barrier voltage.

State preparation (X180 on target) uses a perfect unitary gate via matrix
multiplication; only the exchange interaction is time-evolved through the
ODE solver — same hybrid approach as test_16a and test_16.

The analysis test validates that the FFT-based 2π period extraction
correctly identifies the oscillation frequency at each amplitude and
that the extracted (V, T_2π) curve is physically consistent.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

from virtual_qpu import prepare_bloch_trajectories, plot_bloch_trajectories
from virtual_qpu.dynamics import simulate as vqpu_simulate
from virtual_qpu.operators import expval as vqpu_expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as vqpu_sweep

from calibration_utils.swap_oscillations.analysis import (
    _extract_oscillation_period,
    _fit_exchange_decay_model,
)

from .conftest import (
    ARTIFACTS_BASE,
    QUBIT_PAIR_NAMES,
    DEFAULT_SOLVER,
    DEFAULT_NOISE_STD,
    DEFAULT_PINK_STD,
    DEFAULT_BROWN_STD,
    _colored_noise,
)

NODE_NAME = "18a_swap_oscillations"

# 2D sweep grid (kept moderate for test speed)
N_VOLTAGES = 200
V_MIN = 0.30
V_MAX = 0.45

DURATION_STEP_NS = 1.
DUR_MIN = 0.0
DUR_MAX = 60.0
_DUR_STEPS = int(np.floor((DUR_MAX - DUR_MIN) / DURATION_STEP_NS))
DUR_LAST_SWEPT = DUR_MIN + _DUR_STEPS * DURATION_STEP_NS
N_DURATIONS = _DUR_STEPS + 1

RETURN_TO_INIT_NS = 16.0
BLOCH_N_TIMEPOINTS = 2000

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=250_000)}

# ---------------------------------------------------------------------------
# Perfect unitary gates (4×4 two-qubit Hilbert space)
# ---------------------------------------------------------------------------
_I2 = jnp.eye(2, dtype=jnp.complex64)

_Rx_pi = jnp.array(
    [[0.0, -1j], [-1j, 0.0]], dtype=jnp.complex64
)

# X180 on target qubit (mode 0): Rx(π) ⊗ I₂  →  |00⟩ → |10⟩
_U_PI_Q0 = jnp.kron(_Rx_pi, _I2)

# Single-qubit |1⟩⟨1| projector (used with device.embed for per-qubit readout)
_PROJ_1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def _make_exchange_schedule(exchange_voltage, exchange_duration):
    """Exchange-only schedule: exchange pulse + return-to-init idle."""
    sched = Schedule()
    ref_ex = sched.play(
        SquarePulse(
            duration=exchange_duration,
            amplitude=exchange_voltage,
            frequency=0.0,
        ),
        channel="exchange_0_1",
    )
    sched.play(
        SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
        channel="exchange_0_1",
        after=[ref_ex],
    )
    return sched.resolve()


# ---------------------------------------------------------------------------
# Main test: 2D swap-oscillation sweep + analysis
# ---------------------------------------------------------------------------

@pytest.mark.analysis
def test_18a_swap_oscillations_analysis(
    ld_device,
    analysis_runner,
    save_analysis_plot,
):
    """virtual_qpu swap oscillations: 2D exchange sweep, per-qubit readout.

    Prepares |10⟩ (target excited), time-evolves the exchange Hamiltonian,
    and reads out ⟨|1⟩⟨1|⟩ on each qubit independently.  Then runs the
    FFT-based analysis to extract 2π oscillation periods and validates
    the results against physics expectations.
    """
    device = ld_device

    voltages_np = np.linspace(V_MIN, V_MAX, N_VOLTAGES, dtype=np.float64)
    voltages = jnp.asarray(voltages_np, dtype=jnp.float32)

    duration_grid = DUR_MIN + DURATION_STEP_NS * np.arange(
        N_DURATIONS, dtype=np.float64
    )
    durations = jnp.asarray(duration_grid, dtype=jnp.float32)

    # Prepare initial state: X180 on target → |10⟩
    psi0 = device.ground_state()
    psi_prepared = _U_PI_Q0 @ psi0

    # Per-qubit observables
    obs_target = dq.asqarray(device.embed(_PROJ_1, mode=0))
    obs_control = dq.asqarray(device.embed(_PROJ_1, mode=1))

    # ── 2D sweep (voltage × duration) ─────────────────────────────
    jump_ops = device.collapse_operators() if DEFAULT_SOLVER == "me" else None

    def _sweep_2d(observable, seed):
        def _inner(**kwargs):
            resolved = _make_exchange_schedule(**kwargs)
            H_t = device.hamiltonian(resolved)
            dur = jnp.asarray(kwargs["exchange_duration"], dtype=jnp.float32)
            ts = jnp.stack([
                jnp.float32(0.0),
                dur + jnp.float32(RETURN_TO_INIT_NS),
            ])
            sol = vqpu_simulate(
                H_t, psi_prepared, ts, solver=DEFAULT_SOLVER,
                jump_ops=jump_ops, options=_SOLVER_KW,
            )
            return vqpu_expval(sol.states, observable)

        result = np.asarray(vqpu_sweep(
            _inner,
            exchange_voltage=voltages,
            exchange_duration=durations,
        ))
        rng = np.random.default_rng(seed=seed)
        if DEFAULT_NOISE_STD > 0:
            result = result + rng.normal(0, DEFAULT_NOISE_STD, size=result.shape)
        if DEFAULT_PINK_STD > 0 or DEFAULT_BROWN_STD > 0:
            result = result + _colored_noise(rng, result.shape, DEFAULT_PINK_STD, DEFAULT_BROWN_STD)
        return np.clip(result, 0.0, 1.0)

    r_control = _sweep_2d(obs_control, seed=42)
    r_target = _sweep_2d(obs_target, seed=43)

    # Take final time point → shape (n_voltages, n_durations)
    data_control = np.asarray(r_control[..., -1], dtype=np.float64)
    data_target = np.asarray(r_target[..., -1], dtype=np.float64)
    assert data_control.shape == (N_VOLTAGES, N_DURATIONS)
    assert data_target.shape == (N_VOLTAGES, N_DURATIONS)

    # ── Build dataset matching node format ─────────────────────────
    qp_name = QUBIT_PAIR_NAMES[0]
    ds_raw = xr.Dataset(
        {
            f"state_control_{qp_name}": xr.DataArray(
                data_control,
                dims=["exchange_amplitude", "exchange_duration"],
                coords={
                    "exchange_amplitude": voltages_np,
                    "exchange_duration": np.asarray(duration_grid, dtype=np.float64),
                },
                attrs={"long_name": "control qubit state", "units": ""},
            ),
            f"state_target_{qp_name}": xr.DataArray(
                data_target,
                dims=["exchange_amplitude", "exchange_duration"],
                coords={
                    "exchange_amplitude": voltages_np,
                    "exchange_duration": np.asarray(duration_grid, dtype=np.float64),
                },
                attrs={"long_name": "target qubit state", "units": ""},
            ),
        },
    )

    # ── Run through analysis_runner ────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_exchange_amplitude": float(V_MIN),
            "max_exchange_amplitude": float(V_MAX),
            "amplitude_step": float((V_MAX - V_MIN) / max(N_VOLTAGES - 1, 1)),
            "min_exchange_duration_in_ns": int(DUR_MIN),
            "max_exchange_duration_in_ns": int(np.ceil(DUR_LAST_SWEPT)) + 1,
            "duration_step_in_ns": int(DURATION_STEP_NS),
            "snr_threshold": 100.0,
            "analysis_role": "best",
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "figure" in node.results, "plot_data should produce a figure"

    # ── Physics sanity checks ──────────────────────────────────────
    # At t=0 (shortest duration, lowest voltage): target starts excited,
    # control starts in ground. Weak exchange at low V barely changes this.
    assert data_target[0, 0] > 0.7, "target should start near P≈1"
    assert data_control[0, 0] < 0.3, "control should start near P≈0"

    # At higher voltages, exchange oscillations should appear:
    # the standard deviation along the duration axis should be nonzero
    # for the top voltage rows.
    high_v_rows = data_target[-3:, :]
    assert np.std(high_v_rows) > 0.05, (
        "expect visible oscillations at high exchange voltage"
    )

    # ── Analysis result checks ─────────────────────────────────────
    assert "fit_results" in node.results, "analyse_data should produce fit_results"
    assert "ds_fit" in node.results, "analyse_data should produce ds_fit"

    fit = node.results["fit_results"][qp_name]
    assert fit["success"], f"analysis should succeed; got {fit}"
    assert fit["n_valid"] > 0, "should have valid amplitudes"
    assert len(fit["amplitudes_valid"]) == fit["n_valid"]
    assert len(fit["t_2pi_valid"]) == fit["n_valid"]
    assert len(fit["snr_valid"]) == fit["n_valid"]
    assert len(fit["role_selected"]) == fit["n_valid"]
    for role in fit["role_selected"]:
        assert role in ("control", "target", "difference"), (
            f"unexpected role '{role}'"
        )

    assert np.isfinite(fit["best_amplitude"]), "best amplitude should be finite"
    assert np.isfinite(fit["best_t_2pi"]), "best T_2π should be finite"
    assert fit["best_t_2pi"] > 0, "T_2π should be positive"
    assert fit["best_snr"] >= 50.0, "best SNR should exceed threshold"

    # ds_fit should contain per-amplitude arrays
    ds_fit = node.results["ds_fit"]
    assert f"t_2pi_{qp_name}" in ds_fit.data_vars
    assert f"snr_{qp_name}" in ds_fit.data_vars
    assert f"valid_{qp_name}" in ds_fit.data_vars

    # Physics: at high amplitudes, the oscillation period should be shorter
    # (exchange coupling increases with barrier voltage).
    amps_valid = np.array(fit["amplitudes_valid"])
    t_2pi_valid = np.array(fit["t_2pi_valid"])
    if len(amps_valid) >= 5:
        low_mask = amps_valid < np.median(amps_valid)
        high_mask = amps_valid > np.median(amps_valid)
        if low_mask.sum() > 0 and high_mask.sum() > 0:
            mean_t_low = np.mean(t_2pi_valid[low_mask])
            mean_t_high = np.mean(t_2pi_valid[high_mask])
            assert mean_t_high < mean_t_low, (
                f"T_2π should decrease with amplitude: "
                f"low-V mean={mean_t_low:.0f} ns, high-V mean={mean_t_high:.0f} ns"
            )

    # ── Polynomial model checks ─────────────────────────────────────
    assert fit["model_fit_success"], (
        f"polynomial fit should converge; got {fit}"
    )
    m = fit["exchange_decay_model"]
    assert m.get("type") == "polynomial", f"model type should be 'polynomial'; got {m}"
    assert "coeffs" in m and "degree" in m, f"model keys missing: {m}"
    assert m["degree"] >= 1, f"degree should be >= 1; got {m['degree']}"
    assert len(m["coeffs"]) == m["degree"] + 1, (
        f"coeffs length {len(m['coeffs'])} != degree+1 ({m['degree']+1})"
    )

    t_predicted = np.polyval(m["coeffs"], amps_valid)
    rel_error = np.abs(t_predicted - t_2pi_valid) / t_2pi_valid
    assert np.median(rel_error) < 0.15, (
        f"model median relative error {np.median(rel_error):.2%} too large"
    )

    # Verify the model was stored on the CZ macro after update_state
    qubit_pair = node.namespace["qubit_pairs"][0]
    cz_macro = qubit_pair.macros.get("cz")
    if cz_macro is not None:
        stored_model = getattr(cz_macro, "exchange_decay_model", None)
        if stored_model is not None:
            assert stored_model["coeffs"] == m["coeffs"]
            assert stored_model["degree"] == m["degree"]
            t_eval = np.polyval(stored_model["coeffs"], fit["best_amplitude"])
            assert abs(t_eval - fit["best_t_2pi"]) / fit["best_t_2pi"] < 0.15, (
                f"model T_2π({fit['best_amplitude']:.3f}) = {t_eval:.1f} ns "
                f"differs from FFT T_2π = {fit['best_t_2pi']:.1f} ns"
            )

    # ── Save raw-data figure as extra artifact ─────────────────────
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from calibration_utils.swap_oscillations import plot_swap_oscillations

    qp_stub = SimpleNamespace(name=qp_name)
    fig_raw = plot_swap_oscillations(
        ds_raw, [qp_stub], ds_fit=ds_fit, fit_results=node.results["fit_results"],
    )
    save_analysis_plot(fig_raw, artifacts_dir, "raw_swap_oscillations.png")
    plt.close(fig_raw)
    assert (artifacts_dir / "raw_swap_oscillations.png").exists()


# ---------------------------------------------------------------------------
# Unit test: FFT period extraction on synthetic sinusoid
# ---------------------------------------------------------------------------

@pytest.mark.analysis
def test_extract_oscillation_period_synthetic():
    """Validate FFT period extraction on a known sinusoidal signal."""
    true_period = 200.0  # ns
    dt = 2.0  # ns
    n_points = 500
    t = np.arange(n_points) * dt
    signal = 0.5 + 0.4 * np.cos(2 * np.pi * t / true_period)

    period, snr, ok = _extract_oscillation_period(signal, dt, snr_threshold=3.0)
    assert ok, f"extraction should succeed; SNR={snr:.1f}"
    assert abs(period - true_period) / true_period < 0.05, (
        f"extracted period {period:.1f} ns differs from true {true_period:.1f} ns"
    )
    assert snr > 10, f"SNR should be high for clean sinusoid; got {snr:.1f}"


@pytest.mark.analysis
def test_fit_exchange_decay_model_synthetic():
    """Validate the polynomial fit on a known T_2π(V) curve."""
    true_coeffs = np.array([-8000.0, 6000.0, -1500.0, 200.0])  # degree 3
    v = np.linspace(0.15, 0.45, 30)
    t_true = np.polyval(true_coeffs, v)
    rng = np.random.default_rng(seed=77)
    t_noisy = t_true + rng.normal(0, 2.0, size=t_true.shape)

    model, ok = _fit_exchange_decay_model(v, t_noisy)
    assert ok, "fit should converge on clean polynomial data"
    assert model["type"] == "polynomial"
    assert "coeffs" in model and "degree" in model
    t_predicted = np.polyval(model["coeffs"], v)
    rel_error = np.abs(t_predicted - t_true) / np.abs(t_true)
    assert np.median(rel_error) < 0.05, (
        f"model median relative error {np.median(rel_error):.2%} too large"
    )


@pytest.mark.analysis
def test_extract_oscillation_period_noise_only():
    """FFT extraction should fail on pure white noise."""
    rng = np.random.default_rng(seed=99)
    signal = rng.normal(0.5, 0.1, size=200)
    dt = 2.0

    _period, _snr, ok = _extract_oscillation_period(signal, dt, snr_threshold=10.0)
    assert not ok, "extraction should fail on pure noise"


# ---------------------------------------------------------------------------
# Bloch-sphere trajectories at representative exchange durations/voltages
# ---------------------------------------------------------------------------

BLOCH_EXCHANGE_VOLTAGES = [0.20, 0.28, 0.33]
BLOCH_EXCHANGE_DURATION = 400.0


@pytest.mark.analysis
def test_18a_bloch_sphere_trajectories(ld_device):
    """Bloch-sphere trajectories for the swap oscillation.

    For each representative exchange voltage at a fixed duration, only the
    exchange interaction is time-evolved starting from |10⟩. The Bloch
    trajectories show the population transfer between target and control.
    """
    device = ld_device
    psi0 = device.ground_state()
    psi_prepared = _U_PI_Q0 @ psi0

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for ex_voltage in BLOCH_EXCHANGE_VOLTAGES:
        resolved_ex = _make_exchange_schedule(ex_voltage, BLOCH_EXCHANGE_DURATION)
        H_ex = device.hamiltonian(resolved_ex)
        ex_total = float(resolved_ex.total_duration)

        bloch_jump_ops = device.collapse_operators() if DEFAULT_SOLVER == "me" else None
        tsave = jnp.linspace(0.0, ex_total, BLOCH_N_TIMEPOINTS)
        result = vqpu_simulate(
            H_ex, psi_prepared, tsave, solver=DEFAULT_SOLVER,
            jump_ops=bloch_jump_ops, options=_SOLVER_KW,
        )

        bloch_labels = ["target (q0)", "control (q1)"]
        bloch = prepare_bloch_trajectories(
            tsave, result.states, device=device, labels=bloch_labels,
        )
        fig, _ = plot_bloch_trajectories(
            [bloch],
            backend="matplotlib",
            title=(
                f"Swap oscillation Bloch trajectories "
                f"\u2014 V={ex_voltage:.2f} V, t={BLOCH_EXCHANGE_DURATION:.0f} ns"
            ),
            series_labels=["|10\u27E9 initial"],
        )
        out_path = artifacts_dir / f"bloch_swap_V{ex_voltage:.2f}_t{BLOCH_EXCHANGE_DURATION:.0f}ns.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        assert out_path.exists(), f"Bloch plot not created: {out_path}"
