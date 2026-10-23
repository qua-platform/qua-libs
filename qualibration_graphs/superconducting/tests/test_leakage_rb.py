"""Tests for the shared leakage RB helper (nodes 37a / 37b with measure_leakage)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.common_utils.leakage_rb import (  # noqa: E402
    LeakageFit,
    correct_populations,
    fit_leakage_decay,
    interleaved_gate_leakage,
    single_qubit_leakage,
    two_qubit_leakage,
)


def _model(m, l1, l2, p0=0.0):
    lam = 1.0 - l1 - l2
    p_inf = l1 / (l1 + l2)
    return p_inf + (p0 - p_inf) * lam**m


def test_fit_recovers_leakage_and_seepage_when_saturated():
    depths = np.array([0, 5, 10, 20, 40, 80, 160, 320, 640])
    leaked = _model(depths, 3e-3, 2e-2)
    fit = fit_leakage_decay(depths, leaked, np.full(depths.shape, 1e-4))
    assert fit.success and fit.method == "exponential"
    assert fit.l1 == pytest.approx(3e-3, rel=0.02)
    assert fit.l2 == pytest.approx(2e-2, rel=0.02)


def test_fit_gives_l1_from_the_slope_when_unsaturated():
    rng = np.random.default_rng(1)
    depths = np.array([0, 2, 4, 8, 16, 32, 64])
    leaked = _model(depths, 2e-4, 1e-2) + rng.normal(0.0, 5e-5, depths.size)
    fit = fit_leakage_decay(depths, leaked, np.full(depths.shape, 5e-5))
    assert fit.success
    assert fit.l1 == pytest.approx(2e-4, rel=0.3)


def test_fit_needs_three_depths():
    fit = fit_leakage_decay([0, 1], [0.0, 0.1])
    assert not fit.success
    assert np.all(np.isnan(fit.curve([0, 1])))


def test_correct_populations_undoes_the_confusion_matrix():
    confusion = np.array([[0.97, 0.02, 0.01], [0.05, 0.93, 0.02], [0.02, 0.08, 0.90]])
    true = np.array([[0.6, 0.35, 0.05], [0.2, 0.7, 0.1]])
    corrected = correct_populations(true @ confusion, confusion)
    np.testing.assert_allclose(corrected, true, atol=1e-12)


def test_correct_populations_passes_through_without_a_matrix():
    measured = np.array([0.5, 0.4, 0.1])
    np.testing.assert_array_equal(correct_populations(measured, None), measured)


def test_two_qubit_leakage_reads_the_3c_plus_t_encoding():
    # 10 shots along axis 0: control f in 3 shots, target f in 2, both in 1.
    states = np.array([6, 7, 8, 2, 5, 0, 1, 3, 4, 0])
    leaked = two_qubit_leakage(states, shots_axis=0)
    assert leaked["control"] == pytest.approx(0.3)
    assert leaked["target"] == pytest.approx(0.3)
    assert leaked["pair"] == pytest.approx(0.5)


def test_two_qubit_leakage_corrects_each_qubit():
    confusion = np.array([[0.95, 0.03, 0.02], [0.04, 0.92, 0.04], [0.02, 0.06, 0.92]])
    rng = np.random.default_rng(0)
    shots = 200_000
    true_control = rng.choice(3, shots, p=[0.7, 0.28, 0.02])
    measured_control = np.array([rng.choice(3, p=confusion[s]) for s in true_control[:20_000]])
    states = 3 * measured_control + 0  # target always read g
    leaked = two_qubit_leakage(states, shots_axis=0, confusion_control=confusion)
    assert leaked["control"] == pytest.approx(0.02, abs=0.005)


def test_single_qubit_leakage():
    states = np.array([[0, 1, 2, 2], [0, 0, 0, 2]])
    np.testing.assert_allclose(single_qubit_leakage(states, shots_axis=1), [0.5, 0.25])


def test_interleaved_gate_leakage_subtracts_the_reference():
    interleaved = LeakageFit(l1=5e-4, l1_stderr=3e-5, l2=2e-2, l2_stderr=1e-3, success=True)
    gate = interleaved_gate_leakage(interleaved, 2e-4, 4e-5, 1.5e-2, 1e-3)
    assert gate.l1 == pytest.approx(3e-4)
    assert gate.l1_stderr == pytest.approx(5e-5)
    assert gate.l2 == pytest.approx(5e-3)


def test_interleaved_gate_leakage_without_a_reference_is_empty():
    interleaved = LeakageFit(l1=5e-4, l1_stderr=3e-5, success=True)
    gate = interleaved_gate_leakage(interleaved, None)
    assert np.isnan(gate.l1)
    assert gate.to_jsonable()["l1"] is None
