"""Analysis test for 14_single_qubit_randomized_benchmarking using virtual_qpu.

Two virtual qubits with different T1/T2 coherence times are simulated by
building the full pulse schedule for each RB circuit and solving the Lindblad
master equation end-to-end.  Each Clifford is decomposed into native gates
whose pulses are appended sequentially to a ``Schedule``; virtual Z gates are
compiled out as accumulated phase offsets on subsequent physical pulses.

Two datasets are produced (``state_virtual_dot_1``, ``state_virtual_dot_2``)
and fed jointly into the analysis pipeline.  The test verifies that the qubit
with shorter T1/T2 shows a lower depolarising parameter alpha (faster RB decay).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    CAYLEY,
    INVERSES,
    decomposition,
)
from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters

from .conftest import (
    DEFAULT_DRIVE_AMP_GHZ,
    _VIRTUAL_QPU_AVAILABLE,
    LossDiVincenzoDevice,
)
from .quam_factory import create_minimal_quam

# ── conditional virtual_qpu imports ───────────────────────────────────────────
try:
    import jax.numpy as jnp
    from virtual_qpu.pulse import GaussianIQPulse
    from virtual_qpu.schedule import Schedule
    from virtual_qpu.dynamics import simulate as _vqpu_simulate
    from quantum_dots.params import LossDiVincenzoParams, MU_B_OVER_H
except ImportError:
    pass  # gracefully skipped below when _VIRTUAL_QPU_AVAILABLE is False

# =============================================================================
# Constants
# =============================================================================

NODE_NAME = "14_single_qubit_randomized_benchmarking"
MAX_CIRCUIT_DEPTH = 64  # log-scale → depths [2, 4, 8, 16, 32, 64]
NUM_CIRCUITS = 5
NUM_SHOTS = 100
RNG_SEED = 99

# Coherence times (ns) for the two synthetic qubits
T1_HIGH, T2_HIGH = 10_000.0, 5_000.0  # qubit A: long-lived
T1_LOW, T2_LOW = 400.0, 200.0  # qubit B: short-lived

# Physical gate specifications: (amplitude_factor, base_phase) relative to amp_pi.
# The virtual_qpu drive Hamiltonian is H = Ω/2·(cos φ σ_x − sin φ σ_y),
# so phase = −π/2 drives along +Y and phase = +π/2 drives along −Y.
_PHYSICAL_GATE_SPECS: dict[str, tuple[float, float]] = {
    "x90": (0.5, 0.0),
    "x180": (1.0, 0.0),
    "xm90": (0.5, np.pi),
    "y90": (0.5, -np.pi / 2),
    "y180": (1.0, -np.pi / 2),
    "ym90": (0.5, np.pi / 2),
}

_VIRTUAL_Z_PHASES: dict[str, float] = {
    "z90": np.pi / 2,
    "z180": np.pi,
    "zm90": -np.pi / 2,
}

# =============================================================================
# Device factory
# =============================================================================


def _make_single_qubit_device(t1: float, t2: float) -> "LossDiVincenzoDevice":
    """Create a 1-qubit LossDiVincenzoDevice with the given T1/T2 (ns)."""
    params = LossDiVincenzoParams(
        n_qubits=1,
        g_factors=[2.0],
        magnetic_field=10.0 / (2.0 * MU_B_OVER_H),  # f_qubit ≈ 10 GHz
        exchange_models=[],
        ref_freqs=None,
        frame="rot",
        use_rwa=True,
        t1=[t1],
        t2=[t2],
    )
    return LossDiVincenzoDevice(params=params)


# =============================================================================
# Pulse-schedule simulation
# =============================================================================


def _build_rb_schedule(
    cliff_sequence: list[int],
    amp_pi: float,
    qubit_freq_ghz: float,
    gate_duration: float,
    gate_sigma: float,
) -> tuple["Schedule", float]:
    """Build a pulse Schedule for a full RB Clifford sequence.

    Virtual Z gates are compiled out as accumulated phase offsets on
    subsequent physical pulses — exactly how the real hardware handles
    frame rotations.

    Returns (schedule, total_duration_ns).
    """
    sched = Schedule()
    z_phase = 0.0
    n_physical = 0

    for cliff_idx in cliff_sequence:
        for gate_name in decomposition(int(cliff_idx)):
            if gate_name in _VIRTUAL_Z_PHASES:
                z_phase += _VIRTUAL_Z_PHASES[gate_name]
            else:
                amp_factor, base_phase = _PHYSICAL_GATE_SPECS[gate_name]
                sched.play(
                    GaussianIQPulse(
                        duration=gate_duration,
                        amplitude=float(amp_pi * amp_factor),
                        frequency=qubit_freq_ghz,
                        phase=float(base_phase + z_phase),
                        sigma=gate_sigma,
                    ),
                    channel="drive_q0",
                )
                n_physical += 1

    return sched, n_physical * gate_duration


def _simulate_rb_survival(
    device: "LossDiVincenzoDevice",
    cliff_sequence: list[int],
    amp_pi: float,
    qubit_freq_ghz: float,
    gate_duration: float,
    gate_sigma: float,
) -> float:
    """Simulate one full RB circuit and return survival probability P(|0⟩)."""
    sched, total_duration = _build_rb_schedule(
        cliff_sequence,
        amp_pi,
        qubit_freq_ghz,
        gate_duration,
        gate_sigma,
    )
    if total_duration == 0.0:
        return 1.0

    resolved = sched.resolve()
    H = device.hamiltonian(resolved)
    psi0 = device.ground_state()
    jump_ops = device.collapse_operators()
    tsave = jnp.array([0.0, total_duration])
    sol = _vqpu_simulate(H, psi0, tsave, solver="me", jump_ops=jump_ops)
    rho_final = np.asarray(sol.states[-1])
    return float(np.real(rho_final[0, 0]))


# =============================================================================
# RB dataset generation via full pulse-schedule simulation
# =============================================================================


def _generate_rb_ds(
    device: "LossDiVincenzoDevice",
    amp_pi: float,
    qubit_freq_ghz: float,
    gate_duration: float,
    gate_sigma: float,
    depths: np.ndarray,
    num_circuits: int,
    num_shots: int,
    seed: int,
    state_var: str,
) -> xr.Dataset:
    """Generate an RB dataset by simulating full pulse sequences.

    Mirrors the incremental Clifford composition algorithm of the QUA PPU:
    one random sequence of maximum length is generated per circuit; shorter
    depths are prefixes of that sequence with the appropriate recovery
    Clifford appended.  Each complete sequence is turned into a pulse
    Schedule and simulated end-to-end via the Lindblad solver.
    """
    rng = np.random.default_rng(seed)
    max_random = int(depths[-1]) - 1
    state_data = np.zeros((num_circuits, len(depths)))

    for ci in range(num_circuits):
        random_seq = rng.integers(0, 24, size=max_random)
        total_clifford = 0
        prev_count = 0

        for di, depth in enumerate(depths):
            num_random = int(depth) - 1
            for i in range(prev_count, num_random):
                total_clifford = CAYLEY[int(random_seq[i])][total_clifford]
            prev_count = num_random

            inverse_cliff = INVERSES[total_clifford]
            full_seq = list(random_seq[:num_random]) + [inverse_cliff]

            p0 = _simulate_rb_survival(
                device,
                full_seq,
                amp_pi,
                qubit_freq_ghz,
                gate_duration,
                gate_sigma,
            )
            state_data[ci, di] = rng.binomial(num_shots, np.clip(p0, 0.0, 1.0)) / num_shots

    return xr.Dataset(
        {state_var: xr.DataArray(state_data, dims=["circuit", "depth"])},
        coords={"circuit": np.arange(num_circuits), "depth": depths},
    )


# =============================================================================
# Test
# =============================================================================


@pytest.mark.analysis
def test_14_single_qubit_rb_virtual_qpu_analysis(rabi_chevron_calibration, analysis_runner):
    """RB via virtual_qpu full pulse-schedule Lindblad simulation.

    Verifies that the qubit with shorter coherence times (lower T1/T2) produces
    a lower depolarising parameter alpha after exponential decay fitting.
    """
    if not _VIRTUAL_QPU_AVAILABLE:
        pytest.skip("virtual_qpu (dynamiqs) not installed — skipping Lindblad RB test")

    # 1. Pulse parameters from the rabi chevron calibration
    amp_pi = DEFAULT_DRIVE_AMP_GHZ
    gate_duration = rabi_chevron_calibration["optimal_duration"]
    gate_sigma = gate_duration / 5
    qubit_freq_ghz = rabi_chevron_calibration["optimal_frequency"] * 1e-9

    # 2. Resolve QuAM qubit names (Q1 → virtual_dot_1, Q2 → virtual_dot_2)
    machine = create_minimal_quam()
    qubit_name_1 = machine.qubits["Q1"].name  # "virtual_dot_1"
    qubit_name_2 = machine.qubits["Q2"].name  # "virtual_dot_2"

    # 3. Depth schedule
    params = Parameters(
        max_circuit_depth=MAX_CIRCUIT_DEPTH,
        log_scale=True,
        num_circuits_per_length=NUM_CIRCUITS,
        num_shots=NUM_SHOTS,
    )
    depths = params.get_depths()

    # 4. Build two single-qubit devices with different coherence times
    device_a = _make_single_qubit_device(T1_HIGH, T2_HIGH)  # long coherence
    device_b = _make_single_qubit_device(T1_LOW, T2_LOW)  # short coherence

    # 5. Generate RB datasets by simulating full pulse sequences
    ds_a = _generate_rb_ds(
        device_a,
        amp_pi,
        qubit_freq_ghz,
        gate_duration,
        gate_sigma,
        depths,
        NUM_CIRCUITS,
        NUM_SHOTS,
        RNG_SEED,
        f"state_{qubit_name_1}",
    )
    ds_b = _generate_rb_ds(
        device_b,
        amp_pi,
        qubit_freq_ghz,
        gate_duration,
        gate_sigma,
        depths,
        NUM_CIRCUITS,
        NUM_SHOTS,
        RNG_SEED + 1,
        f"state_{qubit_name_2}",
    )
    ds_raw = xr.merge([ds_a, ds_b])

    # 6. Basic dataset sanity checks
    for var in [f"state_{qubit_name_1}", f"state_{qubit_name_2}"]:
        assert var in ds_raw.data_vars, f"Missing variable {var}"
        assert ds_raw[var].shape == (NUM_CIRCUITS, len(depths))
        assert np.all(ds_raw[var].values >= 0.0)
        assert np.all(ds_raw[var].values <= 1.0)

    # High-coherence qubit should survive better at long depths
    mean_a = ds_raw[f"state_{qubit_name_1}"].mean("circuit").values
    mean_b = ds_raw[f"state_{qubit_name_2}"].mean("circuit").values
    assert mean_a[-1] > mean_b[-1], "High-coherence qubit should have higher mean survival at max depth"

    # 7. Run analysis pipeline on both qubits simultaneously
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        analyse_qubits=["Q1", "Q2"],
        param_overrides={
            "max_circuit_depth": MAX_CIRCUIT_DEPTH,
            "log_scale": True,
            "num_circuits_per_length": NUM_CIRCUITS,
            "num_shots": NUM_SHOTS,
        },
    )

    # 8. Fit result assertions
    assert "fit_results" in node.results, "analyse_data must populate fit_results"
    fit_a = node.results["fit_results"][qubit_name_1]
    fit_b = node.results["fit_results"][qubit_name_2]

    assert fit_a["success"], f"Fit for high-coherence qubit failed: {fit_a}"
    assert fit_b["success"], f"Fit for low-coherence qubit failed: {fit_b}"

    # Both alpha values must be physically valid
    assert 0.0 < fit_a["alpha"] <= 1.0, f"alpha_a = {fit_a['alpha']:.4f}"
    assert 0.0 < fit_b["alpha"] <= 1.0, f"alpha_b = {fit_b['alpha']:.4f}"

    # The qubit with longer T1/T2 must have slower RB decay (higher alpha)
    assert fit_a["alpha"] > fit_b["alpha"], (
        f"Expected alpha_a > alpha_b (high vs low coherence), "
        f"got alpha_a={fit_a['alpha']:.4f}, alpha_b={fit_b['alpha']:.4f}"
    )
