"""Tests for the two-qubit leakage RB analysis (37a / 37b with measure_leakage)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.two_qubit_rb.leakage import (  # noqa: E402
    INTERLEAVED_KEY,
    REFERENCE_KEY,
    analyse_leakage,
    state_entry,
    validate_leakage_run,
)

OPERATION = "cz_unipolar"


def _qubit(name, confusion=None, centers=3):
    resonator = SimpleNamespace(
        operations={"readout_GEF": object()},
        gef_centers=[[0.0, 0.0]] * centers,
        gef_confusion_matrix=confusion,
    )
    return SimpleNamespace(name=name, resonator=resonator)


def _node(fidelity=None, reset_type="thermal"):
    control, target = _qubit("qA2"), _qubit("qA1")
    pair = SimpleNamespace(name="qA2-qA1", qubit_control=control, qubit_target=target)
    machine = SimpleNamespace(
        qubit_pairs={"qA2-qA1": SimpleNamespace(macros={OPERATION: SimpleNamespace(fidelity=fidelity or {})})}
    )
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
    ds_leak, results = analyse_leakage(ds, node, interleaved=False)
    result = results["qA2-qA1"]
    assert result["success"]
    assert result["per_clifford"]["control"]["l1"] == pytest.approx(1e-3, rel=0.25)
    assert result["per_clifford"]["target"]["l1"] == pytest.approx(1e-4, abs=1e-4)
    assert ds_leak.leaked_pair.shape == (1, 6)
    entry = state_entry(result, 12, interleaved=False)
    assert entry["load_id"] == 12 and "per_cz" not in entry


def test_interleaved_run_subtracts_the_stored_reference():
    reference_node, _ = _node()
    _, reference = analyse_leakage(_dataset(1e-3, 1e-4, np.random.default_rng(4)), reference_node, interleaved=False)
    fidelity = {REFERENCE_KEY: state_entry(reference["qA2-qA1"], 7, interleaved=False)}
    node, _ = _node(fidelity)
    _, results = analyse_leakage(_dataset(3e-3, 1e-4, np.random.default_rng(5)), node, interleaved=True)
    result = results["qA2-qA1"]
    assert result["reference_load_id"] == 7
    assert result["per_cz"]["control"]["l1"] == pytest.approx(2e-3, rel=0.35)
    assert INTERLEAVED_KEY == "LeakageRB"


def test_interleaved_run_without_a_reference_reports_per_clifford_only():
    node, _ = _node()
    _, results = analyse_leakage(_dataset(3e-3, 1e-4, np.random.default_rng(6)), node, interleaved=True)
    assert results["qA2-qA1"]["per_cz"] is None
    assert "37a" in results["qA2-qA1"]["per_cz_note"]


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
