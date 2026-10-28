"""Tests for the single-qubit RB analysis (11a / 11b, with and without measure_leakage)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibration_utils.single_qubit_randomized_benchmarking.analysis as analysis  # noqa: E402
from calibration_utils.single_qubit_randomized_benchmarking import (  # noqa: E402
    add_interleaved_gate_error,
    fit_leakage_data,
    fit_raw_data,
    merge_leakage,
)

ALPHA_11A = 0.9973  # reference decay per Clifford
ALPHA_11B = 0.99512  # decay per Clifford + x180 step
GATE_ERROR = (1 - ALPHA_11B / ALPHA_11A) / 2  # standard IRB, d = 2
SHOTS = 50


def _dataset(alpha, depths, rng, sequences=100, l1=0.0):
    """g/e/f populations per sequence: survival 0.5 + 0.5 alpha^m, with P(f) ~ l1 * m leaked from it."""
    depths = np.asarray(depths)
    survival = 0.5 + 0.5 * alpha**depths
    leaked = np.clip(l1 * depths, 0, 1)
    pg = rng.binomial(SHOTS, np.broadcast_to(survival * (1 - leaked), (sequences, depths.size))) / SHOTS
    pf = rng.binomial(SHOTS, np.broadcast_to(leaked, (sequences, depths.size))) / SHOTS
    variables = {"pg": pg, "pe": np.clip(1 - pg - pf, 0, 1), "pf": pf}
    return xr.Dataset(
        {key: (("qubit", "nb_of_sequences", "depths"), value[None]) for key, value in variables.items()},
        coords={"qubit": ["qA3"], "nb_of_sequences": np.arange(sequences), "depths": depths},
    )


def _qubit(gate_fidelity=None, extras=None):
    resonator = SimpleNamespace(gef_confusion_matrix=None)
    return SimpleNamespace(name="qA3", gate_fidelity=gate_fidelity or {}, extras=extras or {}, resonator=resonator)


def _node(qubit=None):
    parameters = SimpleNamespace(
        measure_leakage=True, use_state_discrimination=True, num_shots=SHOTS, reset_type="thermal"
    )
    return SimpleNamespace(namespace={"qubits": [qubit or _qubit()]}, parameters=parameters)


def _interleaved(gate_fidelity, rng, saved=None, monkeypatch=None):
    """Fit an 11b run against ``gate_fidelity``; ``saved`` is what loading the 11a run returns."""
    if monkeypatch is not None:
        monkeypatch.setattr(analysis, "load_saved_ds_fit", saved)
    qubit = _qubit(gate_fidelity)
    ds_fit, results = fit_raw_data(_dataset(ALPHA_11B, np.arange(1, 1001, 20), rng), _node(qubit))
    logs = []
    ds_fit = add_interleaved_gate_error(ds_fit, results, [qubit], log=logs.append, gate_label="x180")
    return ds_fit, results["qA3"], logs


def test_standard_rb_recovers_alpha_and_epc():
    ds_fit, results = fit_raw_data(_dataset(ALPHA_11A, np.arange(1, 1001, 20), np.random.default_rng(0)), _node())
    result = results["qA3"]
    assert result.success and not result.fit_issues
    assert result.alpha == pytest.approx(ALPHA_11A, abs=4 * float(ds_fit.alpha_stderr[0]))
    assert result.error_per_clifford == pytest.approx((1 - ALPHA_11A) / 2, rel=0.05)
    assert result.error_per_gate == pytest.approx(result.error_per_clifford / 1.875)


def test_interleaved_uses_the_saved_11a_run_at_its_own_depths(monkeypatch):
    rng = np.random.default_rng(1)
    saved, _ = fit_raw_data(_dataset(ALPHA_11A, 2 ** np.arange(12), rng), _node())  # log-spaced, as 11a #80
    gate_fidelity = {"averaged_alpha": 0.5, "averaged_load_id": 80}  # the saved run wins over the stored alpha
    ds_fit, result, logs = _interleaved(gate_fidelity, rng, saved=lambda load_id: saved, monkeypatch=monkeypatch)
    assert result.success and not logs
    assert float(ds_fit.reference_alpha[0]) == pytest.approx(float(saved.fit_alpha[0]))
    assert int(ds_fit.reference_load_id[0]) == 80
    np.testing.assert_array_equal(ds_fit.reference_depths[0], 2 ** np.arange(12))
    assert result.gate_error == pytest.approx(GATE_ERROR, abs=4 * result.gate_error_stderr)
    # the reference's own uncertainty is propagated
    _, stored_only, _ = _interleaved({"averaged_alpha": float(saved.fit_alpha[0])}, np.random.default_rng(1))
    assert result.gate_error_stderr > stored_only.gate_error_stderr


def test_interleaved_falls_back_to_the_stored_alpha(monkeypatch):
    def unreadable(load_id):
        raise FileNotFoundError(f"no run {load_id}")

    gate_fidelity = {"averaged_alpha": ALPHA_11A, "averaged_load_id": 99}
    ds_fit, result, logs = _interleaved(gate_fidelity, np.random.default_rng(2), unreadable, monkeypatch)
    assert result.success
    assert "no run 99" in logs[0] and "averaged_alpha" in logs[0]
    assert float(ds_fit.reference_alpha[0]) == ALPHA_11A
    assert np.isnan(ds_fit.reference_amplitude[0]) and np.isnan(ds_fit.reference_depths[0]).all()
    assert result.gate_error == pytest.approx(GATE_ERROR, abs=4 * result.gate_error_stderr)


def test_interleaved_without_reference_reports_no_gate_error():
    ds_fit, result, _ = _interleaved({}, np.random.default_rng(3))
    assert result.success and result.gate_error is None
    assert np.isnan(ds_fit.reference_alpha[0]) and np.isnan(ds_fit.gate_error[0])


def test_interleaved_fails_a_gate_fidelity_above_100_percent():
    ds_fit, result, _ = _interleaved({"averaged_alpha": 0.990}, np.random.default_rng(4))  # stale reference
    assert not result.success and not bool(ds_fit.success[0])
    assert result.gate_error < 0
    assert any("exceeds reference 11a averaged_alpha" in issue for issue in result.fit_issues)


def test_leakage_fit_merges_into_ds_fit():
    l1 = 2e-4
    ds = _dataset(ALPHA_11A, np.array([1, 50, 100, 200, 400]), np.random.default_rng(5), l1=l1)
    node = _node()
    ds_fit, _ = fit_raw_data(ds, node)
    ds_leakage, results = fit_leakage_data(ds, node)
    assert results["qA3"].success
    fit = results["qA3"].per_clifford
    assert fit.l1 == pytest.approx(l1, abs=4 * fit.l1_stderr)

    merged = merge_leakage(ds_fit, ds_leakage)
    assert all(merged[name].equals(ds_fit[name]) for name in ds_fit.data_vars)
    assert {"leaked", "leaked_fit", "leakage_l1", "leakage_success"} <= set(merged.data_vars)
    assert bool(merged.success[0]) and bool(merged.leakage_success[0])  # survival and leakage flags kept apart


def test_merge_leakage_refuses_a_clashing_variable():
    ds_fit = xr.Dataset({"success": ("qubit", [True])}, coords={"qubit": ["qA3"]})
    with pytest.raises(ValueError, match="success"):
        merge_leakage(ds_fit, xr.Dataset({"success": ("qubit", [False])}, coords={"qubit": ["qA3"]}))
