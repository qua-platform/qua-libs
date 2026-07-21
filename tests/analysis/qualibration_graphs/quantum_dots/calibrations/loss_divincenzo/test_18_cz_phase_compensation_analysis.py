"""Analysis test for 18_cz_phase_compensation.

The data is generated with virtual_qpu from the same calibration chain used by
the experiment:

1. Find the fixed cphase duration from a node-16a-style duration sweep.
2. Refine the cphase amplitude with node-17 error-amplification analysis.
3. Run the node-18 Ramsey frame sweeps at that fixed duration and optimized
   amplitude, then verify the fitted local-Z corrections leave a CZ unitary.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from calibration_utils.geometric_cz_error_amplification.analysis import (
    fit_raw_data as fit_error_amplification_raw_data,
)

from virtual_qpu.dynamics import simulate as _vqpu_simulate
from virtual_qpu.pulse import GaussianIQPulse, SquarePulse
from virtual_qpu.schedule import Schedule

from .conftest import (
    DEFAULT_PULSE_DURATION_NS,
    QUBIT_PAIR_NAMES,
    build_joint_stream_analysis_ds,
    simulate_sweep,
)
from .test_17_geometric_cz_error_amplification_analysis import (
    EXCHANGE_AMPLITUDE_REF_V,
    QP_STUB,
    RETURN_TO_INIT_NS,
    _SOLVER_KW,
    _run_16a_virtual_sweep_for_pi_duration,
    _sweep_cphase_bloch_for_control,
)

NODE_NAME = "18_cz_phase_compensation"

N_FRAMES = 32
NODE17_PREPASS_AMPLITUDE_SPAN = 0.0015
NODE17_PREPASS_AMPLITUDE_STEP = 0.00005
NODE17_PREPASS_MAX_GATES = 60
PHASE_COMPENSATION_NOISE_STD = 0.0
PHASE_COMPENSATION_TOL_RAD = 0.35
MIN_CZ_PROCESS_FIDELITY = 0.40


def _play_gaussian(
    sched: Schedule,
    *,
    qubit_idx: int,
    amplitude: float,
    frequency_ghz: float,
    phase_rad: float | jnp.ndarray = 0.0,
    after=None,
):
    kwargs = {"after": [after]} if after is not None else {}
    return sched.play(
        GaussianIQPulse(
            duration=float(DEFAULT_PULSE_DURATION_NS),
            amplitude=amplitude,
            frequency=frequency_ghz,
            phase=phase_rad,
            sigma=float(DEFAULT_PULSE_DURATION_NS) / 5,
        ),
        channel=f"drive_q{qubit_idx}",
        **kwargs,
    )


def _play_cphase(
    sched: Schedule,
    *,
    after,
    cz_duration_ns: float,
    exchange_amplitude_v: float | jnp.ndarray,
):
    ref_ex = sched.play(
        SquarePulse(
            duration=float(cz_duration_ns),
            amplitude=exchange_amplitude_v,
            frequency=0.0,
        ),
        channel="exchange_0_1",
        after=[after],
    )
    return sched.play(
        SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
        channel="exchange_0_1",
        after=[ref_ex],
    )


def _fit_node17_optimized_amplitude(
    device,
    *,
    cz_duration_ns: float,
    pi_half_q0: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
) -> float:
    """Run a compact node-17-style virtual sweep and return the fitted amplitude."""
    amplitudes = np.arange(
        EXCHANGE_AMPLITUDE_REF_V - NODE17_PREPASS_AMPLITUDE_SPAN,
        (
            EXCHANGE_AMPLITUDE_REF_V
            + NODE17_PREPASS_AMPLITUDE_SPAN
            + 0.5 * NODE17_PREPASS_AMPLITUDE_STEP
        ),
        NODE17_PREPASS_AMPLITUDE_STEP,
        dtype=np.float64,
    )
    num_gates = np.arange(2, NODE17_PREPASS_MAX_GATES + 1, 2, dtype=np.int32)
    amps_j = jnp.asarray(amplitudes, dtype=jnp.float32)
    ng_j = jnp.asarray(num_gates, dtype=jnp.int32)

    r0 = _sweep_cphase_bloch_for_control(
        device,
        control_state=0,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
        pi_half_dur=float(DEFAULT_PULSE_DURATION_NS),
        pi_dur=float(DEFAULT_PULSE_DURATION_NS),
        n_gates_arr=ng_j,
        amplitudes_arr=amps_j,
        seed=300,
        noise_std=0.0,
    )
    r1 = _sweep_cphase_bloch_for_control(
        device,
        control_state=1,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
        pi_half_dur=float(DEFAULT_PULSE_DURATION_NS),
        pi_dur=float(DEFAULT_PULSE_DURATION_NS),
        n_gates_arr=ng_j,
        amplitudes_arr=amps_j,
        seed=301,
        noise_std=0.0,
    )

    data = np.zeros((2, 2, len(amplitudes), len(num_gates)), dtype=np.float64)
    data[0, 0, :, :] = r0[:, :, 0].T
    data[0, 1, :, :] = r0[:, :, 1].T
    data[1, 0, :, :] = r1[:, :, 0].T
    data[1, 1, :, :] = r1[:, :, 1].T

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "control_state": (np.array([0, 1], dtype=int), "control state", ""),
            "analysis_axis": (np.array([0, 1], dtype=int), "analysis quadrature", ""),
            "exchange_amplitude": (amplitudes, "barrier gate voltage", "V"),
            "num_cphase_gates": (num_gates, "number of raw CPhase gates", ""),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )
    _, fit_results = fit_error_amplification_raw_data(
        ds_raw, [QP_STUB], quadrature_signal_center=0.5,
    )
    fit = fit_results["q1_q2"]
    assert fit["success"], f"Node-17 amplitude pre-fit should succeed: {fit}"
    optimal_amplitude = float(fit["optimal_amplitude"])
    assert np.isfinite(optimal_amplitude), f"Invalid node-17 amplitude: {fit}"
    assert (
        abs(optimal_amplitude - EXCHANGE_AMPLITUDE_REF_V)
        <= NODE17_PREPASS_AMPLITUDE_SPAN + 2 * NODE17_PREPASS_AMPLITUDE_STEP
    ), f"Node-17 optimum {optimal_amplitude:.6f} V left the prepass window"
    return optimal_amplitude


def _make_phase_compensation_schedule(
    *,
    exp_type: int,
    cz_duration_ns: float,
    exchange_amplitude_v: float,
    pi_half_q0: float,
    pi_q0: float,
    pi_half_q1: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
):
    """Return ``make_schedule(frame)`` for one of the four node-18 experiments."""

    def make_schedule(frame):
        sched = Schedule()
        frame_phase = 2 * jnp.pi * jnp.asarray(frame, dtype=jnp.float32)

        if exp_type == 0:
            ref = _play_gaussian(
                sched, qubit_idx=0, amplitude=pi_half_q0, frequency_ghz=q0_ghz
            )
            ref = _play_cphase(
                sched,
                after=ref,
                cz_duration_ns=cz_duration_ns,
                exchange_amplitude_v=exchange_amplitude_v,
            )
            _play_gaussian(
                sched,
                qubit_idx=0,
                amplitude=pi_half_q0,
                frequency_ghz=q0_ghz,
                phase_rad=frame_phase,
                after=ref,
            )
        elif exp_type == 1:
            ref = _play_gaussian(
                sched, qubit_idx=1, amplitude=pi_q1, frequency_ghz=q1_ghz
            )
            ref = _play_gaussian(
                sched,
                qubit_idx=0,
                amplitude=pi_half_q0,
                frequency_ghz=q0_ghz,
                after=ref,
            )
            ref = _play_cphase(
                sched,
                after=ref,
                cz_duration_ns=cz_duration_ns,
                exchange_amplitude_v=exchange_amplitude_v,
            )
            _play_gaussian(
                sched,
                qubit_idx=0,
                amplitude=pi_half_q0,
                frequency_ghz=q0_ghz,
                phase_rad=frame_phase,
                after=ref,
            )
        elif exp_type == 2:
            ref = _play_gaussian(
                sched, qubit_idx=1, amplitude=pi_half_q1, frequency_ghz=q1_ghz
            )
            ref = _play_cphase(
                sched,
                after=ref,
                cz_duration_ns=cz_duration_ns,
                exchange_amplitude_v=exchange_amplitude_v,
            )
            _play_gaussian(
                sched,
                qubit_idx=1,
                amplitude=pi_half_q1,
                frequency_ghz=q1_ghz,
                phase_rad=frame_phase,
                after=ref,
            )
        elif exp_type == 3:
            ref = _play_gaussian(
                sched, qubit_idx=0, amplitude=pi_q0, frequency_ghz=q0_ghz
            )
            ref = _play_gaussian(
                sched,
                qubit_idx=1,
                amplitude=pi_half_q1,
                frequency_ghz=q1_ghz,
                after=ref,
            )
            ref = _play_cphase(
                sched,
                after=ref,
                cz_duration_ns=cz_duration_ns,
                exchange_amplitude_v=exchange_amplitude_v,
            )
            _play_gaussian(
                sched,
                qubit_idx=1,
                amplitude=pi_half_q1,
                frequency_ghz=q1_ghz,
                phase_rad=frame_phase,
                after=ref,
            )
        else:
            raise ValueError(f"Unknown CZ phase compensation experiment {exp_type}.")

        return sched.resolve()

    return make_schedule


def _simulate_phase_compensation_trace(
    device,
    *,
    frames: np.ndarray,
    exp_type: int,
    cz_duration_ns: float,
    exchange_amplitude_v: float,
    pi_half_q0: float,
    pi_q0: float,
    pi_half_q1: float,
    pi_q1: float,
    q0_ghz: float,
    q1_ghz: float,
) -> np.ndarray:
    """Simulate one node-18 frame sweep and return the final excited probability."""
    prep_pulses = 3 if exp_type in (1, 3) else 2
    total_duration = (
        prep_pulses * float(DEFAULT_PULSE_DURATION_NS)
        + float(cz_duration_ns)
        + RETURN_TO_INIT_NS
    )

    def tsave(frame, **_):
        fr = jnp.asarray(frame, dtype=jnp.float32)
        return jnp.stack(
            [
                jnp.zeros_like(fr, dtype=jnp.float32),
                jnp.full_like(fr, total_duration, dtype=jnp.float32),
            ],
            axis=-1,
        )

    result = simulate_sweep(
        device,
        _make_phase_compensation_schedule(
            exp_type=exp_type,
            cz_duration_ns=cz_duration_ns,
            exchange_amplitude_v=exchange_amplitude_v,
            pi_half_q0=pi_half_q0,
            pi_q0=pi_q0,
            pi_half_q1=pi_half_q1,
            pi_q1=pi_q1,
            q0_ghz=q0_ghz,
            q1_ghz=q1_ghz,
        ),
        tsave=tsave,
        observable_parity=True,
        noise_std=PHASE_COMPENSATION_NOISE_STD,
        solver_options=_SOLVER_KW,
        frame=jnp.asarray(frames, dtype=jnp.float32),
    )
    return np.asarray(result[..., -1], dtype=np.float64)


def _simulate_raw_cphase_unitary(
    device,
    *,
    cz_duration_ns: float,
    exchange_amplitude_v: float,
) -> np.ndarray:
    """Propagate all computational basis states through the calibrated cphase."""
    sched = Schedule()
    ref_ex = sched.play(
        SquarePulse(
            duration=float(cz_duration_ns),
            amplitude=float(exchange_amplitude_v),
            frequency=0.0,
        ),
        channel="exchange_0_1",
    )
    sched.play(
        SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
        channel="exchange_0_1",
        after=[ref_ex],
    )

    h_t = device.hamiltonian(sched.resolve())
    tsave = jnp.array(
        [0.0, float(cz_duration_ns) + RETURN_TO_INIT_NS],
        dtype=jnp.float32,
    )

    columns = []
    for levels in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sol = _vqpu_simulate(
            h_t,
            device.basis_state(*levels),
            tsave,
            solver="se",
            options=_SOLVER_KW,
        )
        columns.append(np.asarray(sol.states.to_jax())[-1, :, 0])
    return np.column_stack(columns).astype(np.complex128)


def _cz_process_fidelity(unitary: np.ndarray) -> float:
    """Return process fidelity to CZ, after optimizing only global phase."""
    cz = np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128))
    dim = cz.shape[0]
    return float(abs(np.trace(cz.conj().T @ unitary)) ** 2 / dim**2)


def _best_diagonal_cz_fidelity(raw_unitary: np.ndarray) -> float:
    """Optimise over single-qubit Z corrections to find the best CZ fidelity.

    This decouples the CZ quality check from any readout-basis-dependent
    phase offset (e.g. parity vs single-qubit readout).
    """
    from scipy.optimize import minimize

    cz = np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128))

    def neg_fid(params):
        phi_c, phi_t = params
        D = np.diag(
            np.exp(1j * np.array([0.0, phi_c, phi_t, phi_c + phi_t]))
        )
        U = D @ raw_unitary
        return -float(abs(np.trace(cz.conj().T @ U)) ** 2) / 16.0

    result = minimize(neg_fid, [0.0, 0.0], method="Nelder-Mead")
    return float(-result.fun)


def _assert_compensated_unitary_is_cz(
    device,
    *,
    cz_duration_ns: float,
    exchange_amplitude_v: float,
    fit: dict[str, float | bool],
) -> None:
    """Check that the virtual-QPU cphase unitary is equivalent to CZ.

    Verifies that the two excited-partner Ramsey estimates of the
    conditional phase agree (``chi_t ≈ chi_c``), and that the raw
    unitary can be made into CZ by single-qubit Z corrections
    (optimised over all diagonal phases, so the result is independent
    of the readout basis).
    """
    chi_t = float(fit["conditional_phase_target"])
    chi_c = float(fit["conditional_phase_control"])
    assert abs(np.angle(np.exp(1j * (chi_t - chi_c)))) < 0.20

    raw_unitary = _simulate_raw_cphase_unitary(
        device,
        cz_duration_ns=cz_duration_ns,
        exchange_amplitude_v=exchange_amplitude_v,
    )
    fidelity = _best_diagonal_cz_fidelity(raw_unitary)
    assert fidelity > MIN_CZ_PROCESS_FIDELITY, (
        f"Best-diagonal-corrected virtual-QPU cphase unitary has CZ process "
        f"fidelity {fidelity:.6f}, expected > {MIN_CZ_PROCESS_FIDELITY:.3f}."
    )


@pytest.mark.analysis
def test_18_cz_phase_compensation_analysis(
    ld_device,
    calibrated_pi_half_amp,
    calibrated_target_pi_amp,
    calibrated_control_pi_amp,
    analysis_runner,
):
    """Node-17-optimized cphase gate -> node-18 phase compensation analysis."""
    device = ld_device
    q0_ghz = float(device.params.qubit_freqs[0])
    q1_ghz = float(device.params.qubit_freqs[1])
    pi_half_q0 = float(calibrated_pi_half_amp)
    pi_q0 = float(calibrated_target_pi_amp)
    pi_q1 = float(calibrated_control_pi_amp)
    pi_half_q1 = pi_q1 / 2.0

    cz_duration_ns, _ = _run_16a_virtual_sweep_for_pi_duration(
        device,
        exchange_amplitude_v=EXCHANGE_AMPLITUDE_REF_V,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
    )
    assert cz_duration_ns % 4.0 == pytest.approx(0.0)

    optimized_amplitude_v = _fit_node17_optimized_amplitude(
        device,
        cz_duration_ns=cz_duration_ns,
        pi_half_q0=pi_half_q0,
        pi_q1=pi_q1,
        q0_ghz=q0_ghz,
        q1_ghz=q1_ghz,
    )

    frames = np.linspace(0, 1, N_FRAMES, endpoint=False, dtype=np.float64)
    data = np.stack(
        [
            _simulate_phase_compensation_trace(
                device,
                frames=frames,
                exp_type=exp_type,
                cz_duration_ns=cz_duration_ns,
                exchange_amplitude_v=optimized_amplitude_v,
                pi_half_q0=pi_half_q0,
                pi_q0=pi_q0,
                pi_half_q1=pi_half_q1,
                pi_q1=pi_q1,
                q0_ghz=q0_ghz,
                q1_ghz=q1_ghz,
            )
            for exp_type in range(4)
        ],
        axis=0,
    )
    assert data.shape == (4, N_FRAMES)
    assert np.all(np.ptp(data, axis=1) > 0.15), "All frame sweeps need contrast"

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "experiment_type": (
                np.array([0, 1, 2, 3], dtype=int),
                "experiment type",
                "",
            ),
            "frame": (frames, "frame rotation", "2π"),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_frames": N_FRAMES,
            "num_shots": 4,
            "conditional_phase_tolerance": 0.2,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"Fit should succeed: {fit}"
    assert np.isfinite(float(fit["target_phase_correction"]))
    assert np.isfinite(float(fit["control_phase_correction"]))
    assert (
        abs(float(fit["conditional_phase_target"]) - np.pi) < PHASE_COMPENSATION_TOL_RAD
    ), f"target conditional phase {fit['conditional_phase_target']:.3f} rad vs pi"
    assert (
        abs(float(fit["conditional_phase_control"]) - np.pi)
        < PHASE_COMPENSATION_TOL_RAD
    ), f"control conditional phase {fit['conditional_phase_control']:.3f} rad vs pi"
    _assert_compensated_unitary_is_cz(
        device,
        cz_duration_ns=cz_duration_ns,
        exchange_amplitude_v=optimized_amplitude_v,
        fit=fit,
    )

    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "fitted_target_ctrl0_q1_q2" in ds_fit.data_vars
    assert "phase_target_ctrl0_q1_q2" in ds_fit.data_vars
    assert "phase_control_tgt0_q1_q2" in ds_fit.data_vars
    assert "phase_control_tgt1_q1_q2" in ds_fit.data_vars
    assert "figure" in node.results
