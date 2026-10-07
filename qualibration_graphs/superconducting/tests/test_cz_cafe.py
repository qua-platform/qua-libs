"""Tests for the CZ CAFE circuits and error-budget fit (node 37c)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.cz_cafe.analysis import (  # noqa: E402
    MAX_DEPTH0_CHI2_RED,
    depth0_consistency,
    fit_cafe_curve,
    model_fidelity,
)
from calibration_utils.cz_cafe.circuits import (  # noqa: E402
    CZ,
    build_circuit_angles,
    compiled_layer,
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


@pytest.mark.parametrize("variant", ["cafe", "decaf"])
@pytest.mark.parametrize("reference", [CZ, fsim_unitary(0.04, 0.03, -0.05)])
def test_switch_tables_reproduce_every_circuit(variant, reference):
    """The tables played by the QUA switch cases (units of 2π) give back every compiled circuit."""
    depths = list(range(0, 10))
    angles = build_circuit_angles([variant], depths, reference)
    preparation = 2 * np.pi * angles.preparation_2pi()
    undo_circuits, undo_index = angles.undo_table_2pi()
    undo_circuits = 2 * np.pi * undo_circuits
    assert undo_index.shape == (1, len(depths), 16)
    cycle = cycle_unitary(reference, variant)
    for idepth, depth in enumerate(depths):
        for istate in range(16):
            undo = undo_circuits[undo_index[0, idepth, istate]]
            p00 = simulate_return_probability(preparation[istate], undo, cycle, depth)
            assert p00 == pytest.approx(1, abs=1e-10)


def test_undo_table_shares_circuits_between_depths():
    # With an ideal CZ, CZ^n is the identity for even n and CZ for odd n: 2 x 16 distinct circuits
    _, index = build_circuit_angles(["cafe"], list(range(0, 17)), CZ).undo_table_2pi()
    assert len(np.unique(index)) == 32
    assert np.array_equal(index[0, 0], index[0, 2])


def _random_unitary(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    return q * (np.diag(r) / abs(np.diag(r)))


def test_depth0_return_probability_is_state_independent_for_any_gate():
    """The n = 0 check in the analysis relies on this: any two-qubit gate error gives the same P(|00>)."""
    rng = np.random.default_rng(3)
    angles = build_circuit_angles(["cafe"], [0], CZ)
    for _ in range(5):
        gate = _random_unitary(rng, 4)
        p = []
        for istate in range(16):
            prep, undo = angles.preparation[istate], angles.undo[0, 0, istate]
            psi = compiled_layer(prep[1]) @ gate @ compiled_layer(prep[0]) @ np.array([1, 0, 0, 0])
            psi = compiled_layer(undo[1]) @ gate @ compiled_layer(undo[0]) @ psi
            p.append(abs(psi[0]) ** 2)
        assert np.ptp(p) == pytest.approx(0, abs=1e-10)


def test_depth0_consistency_flags_state_dependent_data():
    rng = np.random.default_rng(5)
    shots = 100
    consistent = rng.binomial(shots, 0.9, size=16) / shots
    assert depth0_consistency(consistent, shots) < MAX_DEPTH0_CHI2_RED
    # Measured depth-0 data from hardware where the layers were not played as compiled
    measured = np.array([0.71, 0.64, 0.8, 0.68, 0.28, 0.24, 0.21, 0.29, 0.7, 0.75, 0.59, 0.7, 0.32, 0.44, 0.27, 0.37])
    assert depth0_consistency(measured, shots) > MAX_DEPTH0_CHI2_RED


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
