"""Analysis test for 17a_iswap_error_amplification.

The test uses the same virtual-QPU calibration chain as the CZ phase-compensation
test:

1. Find the fixed CPhase duration from a node-16a-style virtual sweep.
2. Refine the CPhase amplitude with the node-17 error-amplification analysis.
3. Build repeated odd-subspace transfer traces from the virtual-QPU raw CPhase
   unitary and verify node 17a extracts the residual iSWAP angle.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import QUBIT_PAIR_NAMES, build_joint_stream_analysis_ds
from .test_17_geometric_cz_error_amplification_analysis import (
    EXCHANGE_AMPLITUDE_REF_V,
    _run_16a_virtual_sweep_for_pi_duration,
)
from .test_18_cz_phase_compensation_analysis import (
    _fit_node17_optimized_amplitude,
    _simulate_raw_cphase_unitary,
)

NODE_NAME = "17a_iswap_error_amplification"

MAX_THETA_RAD = 0.25
# Parity readout (P_odd) differs from single-qubit readout (P_excited)
# by a non-linear mapping in the odd subspace, so the transfer-model fit
# gives a slightly different theta.  0.015 rad accommodates this bias.
THETA_ABS_TOL = 0.015

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


def _cycle_unitary(raw_cphase: np.ndarray, *, dd_axis: int) -> np.ndarray:
    """Return one two-CPhase MEADD cycle unitary for a DD axis."""
    if dd_axis == 0:
        dd = np.kron(_X, _X)
    elif dd_axis == 1:
        # Basis order is target⊗control in the virtual-QPU tests.  The node's
        # Y⊗X branch applies Y on the control and X on the target.
        dd = np.kron(_X, _Y)
    else:
        raise ValueError(f"Unknown DD axis {dd_axis}")
    return dd @ raw_cphase @ dd @ raw_cphase


def _transfer_traces_from_unitary(
    raw_cphase: np.ndarray,
    num_cycles: np.ndarray,
) -> np.ndarray:
    """Return parity traces with dims (initial_state, dd_axis, depth).

    Uses the parity projector P_odd = |01⟩⟨01| + |10⟩⟨10|, which gives 0 for
    parallel spins (↑↑, ↓↓) and 1 for antiparallel spins (↑↓, ↓↑).
    P_odd(ψ) = |ψ[1]|² + |ψ[2]|²  where ψ[1]=⟨01|ψ⟩, ψ[2]=⟨10|ψ⟩.
    """
    data = np.zeros((2, 2, len(num_cycles)), dtype=np.float64)

    initial_states = {
        0: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),  # control excited
        1: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),  # target excited
    }
    for dd_axis in (0, 1):
        cycle = _cycle_unitary(raw_cphase, dd_axis=dd_axis)
        for state_idx, psi0 in initial_states.items():
            for depth_idx, depth in enumerate(num_cycles):
                psi = np.linalg.matrix_power(cycle, int(depth)) @ psi0
                data[state_idx, dd_axis, depth_idx] = float(
                    abs(psi[1]) ** 2 + abs(psi[2]) ** 2
                )
    return data


def _raw_iswap_theta_abs(raw_cphase: np.ndarray) -> float:
    """Extract |theta| from the raw odd-subspace matrix element."""
    odd = raw_cphase[np.ix_([1, 2], [1, 2])]
    sin_theta = float(np.mean([abs(odd[0, 1]), abs(odd[1, 0])]))
    return float(np.arcsin(np.clip(sin_theta, 0.0, 1.0)))


@pytest.mark.analysis
def test_17a_iswap_error_amplification_analysis(
    ld_device,
    calibrated_pi_half_amp,
    calibrated_control_pi_amp,
    analysis_runner,
):
    """Node-17-optimized CPhase gate -> residual iSWAP amplification fit."""
    device = ld_device
    q0_ghz = float(device.params.qubit_freqs[0])
    q1_ghz = float(device.params.qubit_freqs[1])
    pi_half_q0 = float(calibrated_pi_half_amp)
    pi_q1 = float(calibrated_control_pi_amp)

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

    raw_cphase = _simulate_raw_cphase_unitary(
        device,
        cz_duration_ns=cz_duration_ns,
        exchange_amplitude_v=optimized_amplitude_v,
    )
    expected_theta = _raw_iswap_theta_abs(raw_cphase)

    num_cycles = np.arange(0, 97, 4, dtype=np.int32)
    data = _transfer_traces_from_unitary(raw_cphase, num_cycles)
    assert data.shape == (2, 2, len(num_cycles))

    ds_raw = build_joint_stream_analysis_ds(
        coords={
            "initial_state": (
                np.array([0, 1], dtype=int),
                "initial odd-parity state",
                "",
            ),
            "dd_axis": (
                np.array([0, 1], dtype=int),
                "DD axis selector",
                "",
            ),
            "num_cphase_cycles": (
                num_cycles,
                "number of raw CPhase two-cycles",
                "",
            ),
        },
        signal_per_qubit={"q1_q2": data},
        qubit_names=QUBIT_PAIR_NAMES,
    )

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "exchange_amplitude": optimized_amplitude_v,
            "exchange_duration_in_ns": int(cz_duration_ns),
            "num_cycle_repetitions": [int(x) for x in num_cycles],
            "max_theta_rad": MAX_THETA_RAD,
            "min_fit_contrast": 1e-8,
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"]["q1_q2"]
    assert fit["success"], f"Fit should succeed: {fit}"
    assert np.isfinite(float(fit["theta_iswap_abs"]))
    assert float(fit["theta_iswap_abs"]) == pytest.approx(
        expected_theta,
        abs=THETA_ABS_TOL,
    )

    assert "ds_fit" in node.results
    ds_fit = node.results["ds_fit"]
    assert "transfer_x_q1_q2" in ds_fit.data_vars
    assert "transfer_y_q1_q2" in ds_fit.data_vars
    assert "transfer_x_fit_q1_q2" in ds_fit.data_vars
    assert "transfer_y_fit_q1_q2" in ds_fit.data_vars
    assert "figure" in node.results
