"""Tests for the two-qubit leakage RB analysis (37a / 37b with measure_leakage)."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.two_qubit_rb import (  # noqa: E402
    RBMode,
    fit_raw_data,
    leakage_per_cz_entry,
    leakage_reference_entry,
    validate_leakage_run,
)
from calibration_utils.two_qubit_rb.fit_utils import _leaked_populations  # noqa: E402
from calibration_utils.common_utils.leakage_rb import (  # noqa: E402
    fit_leakage,
    mean_and_stderr_over_sequences,
)

OPERATION = "cz_unipolar"


def _qubit(name, confusion=None, centers=3):
    resonator = SimpleNamespace(
        operations={"readout_GEF": object()},
        gef_centers=[[0.0, 0.0]] * centers,
        gef_confusion_matrix=confusion,
    )
    return SimpleNamespace(name=name, resonator=resonator, T1=None, T2echo=None)


def _node(fidelity=None, reset_type="thermal", extras=None):
    control, target = _qubit("qA2"), _qubit("qA1")
    macros = {OPERATION: SimpleNamespace(fidelity=fidelity or {}, extras=extras or {})}
    pair = SimpleNamespace(name="qA2-qA1", qubit_control=control, qubit_target=target, macros=macros)
    machine = SimpleNamespace(qubit_pairs={"qA2-qA1": SimpleNamespace(macros=macros)})
    logs = []
    node = SimpleNamespace(
        namespace={"qubit_pairs": [pair]},
        machine=machine,
        parameters=SimpleNamespace(operation=OPERATION, reset_type=reset_type),
        log=lambda message, **_: logs.append(message),
    )
    return node, logs


def _dataset(l1_control, l1_target, rng, shots=4000, sequences=8):
    """Per-shot states 3*c+t where each qubit leaked with probability ~ l1 * depth."""
    depths = np.array([0, 4, 8, 16, 32, 64])
    states = np.zeros((1, shots, depths.size, sequences), dtype=int)
    for d, depth in enumerate(depths):
        c = (rng.random((shots, sequences)) < l1_control * depth) * 2
        t = (rng.random((shots, sequences)) < l1_target * depth) * 2
        states[0, :, d, :] = 3 * c + t
    return xr.Dataset(
        {"state": (("qubit_pair", "shots", "circuit_depth", "sequence"), states)},
        coords={
            "qubit_pair": ["qA2-qA1"],
            "shots": np.arange(shots),
            "circuit_depth": depths,
            "sequence": np.arange(sequences),
        },
    )


def test_reference_run_finds_the_leaking_qubit():
    node, _ = _node()
    ds = _dataset(1e-3, 1e-4, np.random.default_rng(3))
    ds_leak, results = fit_raw_data(ds, node, mode=RBMode.STANDARD, leakage=True)
    result = results["qA2-qA1"]
    assert result.success
    assert result.per_clifford["control"].l1 == pytest.approx(1e-3, rel=0.25)
    assert result.per_clifford["target"].l1 == pytest.approx(1e-4, abs=1e-4)
    assert ds_leak.leaked.sel(leakage_series="pair").shape == (1, 6)
    json.dumps(asdict(result), allow_nan=False)  # written to the state as is: no NaN


def test_interleaved_run_subtracts_the_stored_reference():
    reference_node, _ = _node()
    _, reference = fit_raw_data(
        _dataset(1e-3, 1e-4, np.random.default_rng(4)), reference_node, mode=RBMode.STANDARD, leakage=True
    )
    entry = leakage_reference_entry(asdict(reference["qA2-qA1"]))
    assert set(entry) == {"pair", "control", "target", "updated_at"}
    assert set(entry["control"]) == {"l1", "l1_stderr", "l2", "l2_stderr"}
    node, _ = _node(extras={"leakage_per_2q_clifford": entry})
    _, results = fit_raw_data(
        _dataset(3e-3, 1e-4, np.random.default_rng(5)), node, mode=RBMode.INTERLEAVED, leakage=True
    )
    result = results["qA2-qA1"]
    assert result.per_cz["control"]["l1"] == pytest.approx(2e-3, rel=0.35)
    assert result.reference_updated_at == entry["updated_at"]
    per_cz_entry = leakage_per_cz_entry(asdict(result))
    assert per_cz_entry["control"] == {
        "l1": result.per_cz["control"]["l1"],
        "l1_stderr": result.per_cz["control"]["l1_stderr"],
    }
    json.dumps([entry, per_cz_entry], allow_nan=False)


def test_interleaved_run_without_a_reference_reports_per_clifford_only():
    node, _ = _node()
    _, results = fit_raw_data(
        _dataset(3e-3, 1e-4, np.random.default_rng(6)), node, mode=RBMode.INTERLEAVED, leakage=True
    )
    assert results["qA2-qA1"].per_cz is None
    assert "37a" in results["qA2-qA1"].per_cz_note
    assert leakage_per_cz_entry(asdict(results["qA2-qA1"])) is None


def test_validation_needs_a_thermal_reset():
    node, _ = _node(reset_type="active")
    with pytest.raises(ValueError, match="thermal"):
        validate_leakage_run(node, node.namespace["qubit_pairs"])


def test_validation_needs_gef_centers_and_warns_without_confusion():
    node, logs = _node()
    validate_leakage_run(node, node.namespace["qubit_pairs"])
    assert "qA1" in logs[0] and "qA2" in logs[0]
    node.namespace["qubit_pairs"][0].qubit_target.resonator.gef_centers = [[0.0, 0.0]] * 2
    with pytest.raises(ValueError, match="qA1: no three calibrated gef_centers"):
        validate_leakage_run(node, node.namespace["qubit_pairs"])


def test_interleaved_survival_of_a_leakage_run_uses_standard_rb():
    from calibration_utils.two_qubit_rb import RBMode, analysis

    # The RB decay does not depend on readout or reset: a leakage 37b compares against StandardRB_alpha.
    fidelity = {"StandardRB_alpha": 0.97, "StandardRB": 0.977}
    node, _ = _node(fidelity)
    node.parameters.measure_leakage = True
    node.parameters.use_input_stream = False
    ds = _dataset(1e-3, 1e-4, np.random.default_rng(10))
    ds_fit, _ = analysis.fit_raw_data(ds, node, mode=RBMode.INTERLEAVED)
    assert float(ds_fit.standard_rb_alpha.sel(qubit_pair="qA2-qA1")) == pytest.approx(0.97)

    node, _ = _node({})
    node.parameters.measure_leakage = True
    with pytest.raises(KeyError, match="StandardRB_alpha"):
        analysis.fit_raw_data(ds, node, mode=RBMode.INTERLEAVED)


# --- Fit and populations ----------------------------------------------------------------------


def _model(m, l1, l2, p0=0.0):
    lam = 1.0 - l1 - l2
    p_inf = l1 / (l1 + l2)
    return p_inf + (p0 - p_inf) * lam**m


def test_fit_recovers_leakage_and_seepage_when_saturated():
    depths = np.array([0, 5, 10, 20, 40, 80, 160, 320, 640])
    fit = fit_leakage(depths, _model(depths, 3e-3, 2e-2), np.full(depths.shape, 1e-4))
    assert fit.success and fit.method == "exponential"
    assert fit.l1 == pytest.approx(3e-3, rel=0.02)
    assert fit.l2 == pytest.approx(2e-2, rel=0.02)


def test_fit_gives_l1_from_the_slope_when_unsaturated():
    rng = np.random.default_rng(1)
    depths = np.array([0, 2, 4, 8, 16, 32, 64])
    leaked = _model(depths, 2e-4, 1e-2) + rng.normal(0.0, 5e-5, depths.size)
    fit = fit_leakage(depths, leaked, np.full(depths.shape, 5e-5))
    assert fit.success
    assert fit.l1 == pytest.approx(2e-4, rel=0.3)


def test_fit_needs_three_depths():
    fit = fit_leakage([0, 1], [0.0, 0.1])
    assert not fit.success
    assert np.all(np.isnan(fit.curve([0, 1])))


def test_an_all_zero_depth_does_not_pin_the_fit():
    # Depth 0 reads exactly 0 in every sequence (as clipped, readout-corrected data does), so its
    # spread across sequences is 0. Without the shot-noise floor it would get infinite weight.
    rng = np.random.default_rng(2)
    depths = np.array([0, 4, 8, 16, 32, 64])
    per_sequence = np.clip(5e-4 * depths[:, None] + rng.normal(0.0, 1e-3, (depths.size, 6)), 0.0, None)
    per_sequence[0] = 0.0
    mean, stderr = mean_and_stderr_over_sequences(per_sequence, shots_per_sequence=2000)
    assert np.all(stderr > 0)
    fit = fit_leakage(depths, mean, stderr)
    assert fit.success
    assert fit.l1 == pytest.approx(5e-4, rel=0.2)


def test_two_qubit_leakage_reads_the_3c_plus_t_encoding():
    # 10 shots: control f in 3 shots, target f in 3, both in 1.
    states = np.array([6, 7, 8, 2, 5, 0, 1, 3, 4, 0])
    leaked = _leaked_populations(states)
    assert leaked["control"] == pytest.approx(0.3)
    assert leaked["target"] == pytest.approx(0.3)
    assert leaked["pair"] == pytest.approx(0.5)


def test_two_qubit_leakage_undoes_the_confusion_matrix():
    confusion = np.array([[0.95, 0.03, 0.02], [0.04, 0.92, 0.04], [0.02, 0.06, 0.92]])
    rng = np.random.default_rng(0)
    true_control = rng.choice(3, 20_000, p=[0.7, 0.28, 0.02])
    measured_control = np.array([rng.choice(3, p=confusion[s]) for s in true_control])
    leaked = _leaked_populations(3 * measured_control, confusion_control=confusion)
    assert leaked["control"] == pytest.approx(0.02, abs=0.005)
    assert _leaked_populations(3 * measured_control)["control"] > 0.03  # uncorrected reads high


def test_high_shallow_leakage_warns_about_readout():
    # Shaped like a real run where qA1's g was read as f: ~1 at depth 0, falling with depth.
    depths = np.array([0, 1, 3, 7, 15, 31, 64])
    leaked = np.array([0.99, 0.96, 0.91, 0.81, 0.62, 0.56, 0.64])
    fit = fit_leakage(depths, leaked, np.full(depths.shape, 0.02))
    assert any("readout or reset" in w for w in fit.fit_warnings)
    assert any("not a leakage build-up" in w for w in fit.fit_warnings)


def test_clean_leakage_has_no_soft_warnings():
    depths = np.array([0, 4, 8, 16, 32, 64])
    fit = fit_leakage(depths, 1e-3 + 5e-4 * depths, np.full(depths.shape, 2e-4))
    assert fit.success
    assert not any("readout" in w for w in fit.fit_warnings)


def test_a_curve_far_from_the_data_fails_the_residual_check():
    depths = np.array([0, 4, 8, 16, 32, 64])
    leaked = np.array([0.0, 0.05, 0.0, 0.05, 0.0, 0.05])  # zigzag no model can follow
    fit = fit_leakage(depths, leaked, np.full(depths.shape, 1e-3))
    assert not fit.success
    assert any("sigma" in issue for issue in fit.fit_issues)


def test_a_rejected_exponential_is_a_warning_when_the_line_is_kept():
    depths = np.array([0, 2, 4, 8, 16, 32, 64])
    fit = fit_leakage(depths, 2e-4 * depths, np.full(depths.shape, 5e-5))
    assert fit.success
    assert fit.fit_issues == ()


def test_a_pair_with_bad_gef_readout_fails():
    # qA1 (target) has its g read as f in most shots: high "leakage" already at depth 0.
    node, _ = _node()
    ds = _dataset(1e-3, 1e-4, np.random.default_rng(8))
    states = ds.state.values
    target_g = (states % 3 == 0) & (np.random.default_rng(9).random(states.shape) < 0.9)
    ds["state"].values[:] = np.where(target_g, states + 2, states)
    _, results = fit_raw_data(ds, node, mode=RBMode.STANDARD, leakage=True)
    result = results["qA2-qA1"]
    assert not result.success
    assert "qA1" in result.failure and "qA2" not in result.failure
