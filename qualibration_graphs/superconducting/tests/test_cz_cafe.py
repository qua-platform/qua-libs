"""Tests for the CZ CAFE circuits and error-budget fit (node 37c)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.cz_cafe.analysis import fit_cafe_curve, model_fidelity  # noqa: E402
from calibration_utils.cz_cafe.circuits import (  # noqa: E402
    CZ,
    build_circuit_angles,
    cycle_unitary,
    fsim_unitary,
    one_cz_preparation,
    ry,
    rz,
    sic_states,
    simulate_return_probability,
)

DEPTHS = list(range(0, 17, 2))


def test_sic_states_form_a_2_design():
    states = sic_states()
    assert np.allclose(np.linalg.norm(states, axis=1), 1)
    # Frame potential of a 2-design in dimension 4 is 2 / (d (d + 1)) = 1/10
    overlaps = np.abs(states.conj() @ states.T) ** 4
    assert overlaps.mean() == pytest.approx(0.1, abs=1e-12)


def test_one_cz_preparation_reaches_every_state():
    for psi in sic_states():
        prepared = one_cz_preparation(psi).unitary()[:, 0]
        assert abs(np.vdot(psi, prepared)) ** 2 == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("variant", ["cafe", "decaf"])
@pytest.mark.parametrize("reference", [CZ, fsim_unitary(0.04, 0.03, -0.05)])
def test_compiled_circuits_return_to_ground_with_reference_gate(variant, reference):
    depths = list(range(0, 10))  # include odd depths
    angles = build_circuit_angles([variant], depths, reference)
    cycle = cycle_unitary(reference, variant)
    for idepth, depth in enumerate(depths):
        for istate in range(16):
            p00 = simulate_return_probability(angles.preparation[istate], angles.undo[0, idepth, istate], cycle, depth)
            assert p00 == pytest.approx(1, abs=1e-10)


def _layer(angles, z_sign=1, y_sign=1):
    def single(a, b, c):
        y90 = ry(y_sign * np.pi / 2)
        return rz(z_sign * a) @ y90 @ rz(z_sign * b) @ y90 @ rz(z_sign * c)

    return np.kron(single(*angles[0]), single(*angles[1]))


def _return_probability(prep, undo, cycle, depth, z_sign=1, y_sign=1):
    psi = np.array([1, 0, 0, 0], dtype=complex)
    psi = _layer(prep[1], z_sign, y_sign) @ CZ @ _layer(prep[0], z_sign, y_sign) @ psi
    psi = np.linalg.matrix_power(cycle, depth) @ psi
    psi = _layer(undo[1], z_sign, y_sign) @ CZ @ _layer(undo[0], z_sign, y_sign) @ psi
    return abs(psi[0]) ** 2


@pytest.mark.parametrize("z_sign,y_sign", [(-1, 1), (1, -1), (-1, -1)])
def test_return_probability_is_independent_of_rotation_sign_conventions(z_sign, y_sign):
    """Flipping the sign of virtual Z or of the y90 axis must not change P(|00>)."""
    angles = build_circuit_angles(["cafe", "decaf"], [0, 3, 6], CZ)
    noisy_gate = fsim_unitary(0.05, 0.04, -0.06)
    for iv, variant in enumerate(["cafe", "decaf"]):
        cycle = cycle_unitary(noisy_gate, variant)
        for idepth, depth in enumerate([0, 3, 6]):
            for istate in range(16):
                prep, undo = angles.preparation[istate], angles.undo[iv, idepth, istate]
                nominal = _return_probability(prep, undo, cycle, depth)
                flipped = _return_probability(prep, undo, cycle, depth, z_sign, y_sign)
                assert flipped == pytest.approx(nominal, abs=1e-12)


def _depolarize(rho, p):
    return (1 - p) * rho + p * np.eye(4) / 4


def _simulate_cafe(gate, p_depol, variant, depths, reference=CZ):
    """Density-matrix simulation of the compiled circuits; every CZ (prep, cycle, undo) is noisy."""
    angles = build_circuit_angles([variant], depths, reference)
    fidelity = []
    for idepth, depth in enumerate(depths):
        p_return = []
        for istate in range(16):
            prep, undo = angles.preparation[istate], angles.undo[0, idepth, istate]
            rho = np.zeros((4, 4), dtype=complex)
            rho[0, 0] = 1
            steps = [_layer(prep[0]), gate, _layer(prep[1])]
            steps += [cycle_unitary(gate, variant)] * depth
            steps += [_layer(undo[0]), gate, _layer(undo[1])]
            for u in steps:
                rho = u @ rho @ u.conj().T
                if u is gate or (variant == "decaf" and u is steps[3] and depth > 0):
                    rho = _depolarize(rho, p_depol)
            p_return.append(rho[0, 0].real)
        fidelity.append(np.mean(p_return))
    return np.array(fidelity)


def _true_budget(gate, p_depol, reference=CZ):
    eps_incoh = 3 * p_depol / 4
    eps_coh = 1 - (4 + abs(np.trace(reference.conj().T @ gate)) ** 2) / 20
    return eps_incoh, eps_coh


@pytest.mark.parametrize(
    "angles,p_depol",
    [((0.0, 0.0, 0.0), 0.01), ((0.03, 0.04, -0.05), 0.0), ((0.02, 0.03, 0.06), 0.008)],
)
def test_fit_recovers_budget_from_simulated_circuits(angles, p_depol):
    gate = fsim_unitary(*angles)
    data = _simulate_cafe(gate, p_depol, "cafe", DEPTHS)
    fr, _ = fit_cafe_curve(np.array(DEPTHS), data, np.full(len(DEPTHS), 1e-3))
    eps_incoh, eps_coh = _true_budget(gate, p_depol)
    assert fr.success, fr.message
    assert fr.incoherent_error == pytest.approx(eps_incoh, abs=1.5e-3)
    assert fr.coherent_error == pytest.approx(eps_coh, abs=1.5e-3)
    assert 1 - fr.fidelity == pytest.approx(eps_incoh + eps_coh, abs=2e-3)


def test_fit_with_characterized_reference_removes_known_coherent_error():
    angles = (0.02, 0.05, -0.04)
    gate = fsim_unitary(*angles)
    data = _simulate_cafe(gate, 0.006, "cafe", DEPTHS, reference=gate)
    fr, _ = fit_cafe_curve(np.array(DEPTHS), data, np.full(len(DEPTHS), 1e-3), reference_gate=gate)
    assert fr.success, fr.message
    assert fr.coherent_error == pytest.approx(0, abs=1e-3)
    assert fr.incoherent_error == pytest.approx(3 * 0.006 / 4, abs=1e-3)


def test_decaf_fit_echoes_out_single_qubit_phase():
    gate = fsim_unitary(0.0, 0.08, 0.0)  # only a single-qubit phase error
    cafe = _simulate_cafe(gate, 0.004, "cafe", DEPTHS)
    decaf = _simulate_cafe(gate, 0.004, "decaf", DEPTHS)
    fr_cafe, _ = fit_cafe_curve(np.array(DEPTHS), cafe, np.full(len(DEPTHS), 1e-3), variant="cafe")
    fr_decaf, _ = fit_cafe_curve(np.array(DEPTHS), decaf, np.full(len(DEPTHS), 1e-3), variant="decaf")
    assert fr_cafe.coherent_error == pytest.approx(_true_budget(gate, 0.004)[1], abs=5e-4)
    assert fr_cafe.coherent_error > 2e-3
    assert fr_decaf.coherent_error == pytest.approx(0, abs=1e-3)


def test_fit_recovers_budget_from_noisy_model_data():
    rng = np.random.default_rng(7)
    depths = np.array(DEPTHS)
    truth = (0.01, 0.03, 0.02, -0.04, 0.02)
    shots = 100 * 16
    clean = model_fidelity(depths, *truth)
    data = rng.binomial(shots, clean) / shots
    sigma = np.sqrt(clean * (1 - clean) / shots)
    fr, _ = fit_cafe_curve(depths, data, sigma)
    eps_incoh, eps_coh = _true_budget(fsim_unitary(*truth[1:4]), truth[0])
    assert fr.success, fr.message
    assert fr.spam == pytest.approx(truth[4], abs=0.01)
    assert fr.incoherent_error == pytest.approx(eps_incoh, abs=3 * fr.incoherent_error_error + 1e-3)
    assert fr.coherent_error == pytest.approx(eps_coh, abs=3 * fr.coherent_error_error + 1e-3)


def test_quadratic_cross_check_matches_at_small_error():
    gate = fsim_unitary(0.01, 0.015, -0.01)
    data = _simulate_cafe(gate, 0.004, "cafe", DEPTHS)
    fr, _ = fit_cafe_curve(np.array(DEPTHS), data, np.full(len(DEPTHS), 1e-3), quadratic_max_depth=4)
    assert fr.quadratic.incoherent_error == pytest.approx(fr.incoherent_error, abs=1e-3)
    assert fr.quadratic.coherent_error == pytest.approx(fr.coherent_error, abs=1e-3)


def test_fit_flags_data_that_does_not_follow_the_model():
    depths = np.array(DEPTHS)
    data = np.where(depths % 4 == 0, 0.95, 0.4)  # alternating, not a CAFE decay
    fr, _ = fit_cafe_curve(depths, data, np.full(len(depths), 1e-3))
    assert not fr.success
    assert fr.message
