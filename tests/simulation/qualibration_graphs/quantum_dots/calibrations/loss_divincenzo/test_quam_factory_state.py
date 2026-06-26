"""Regression tests for the wiring-based QuAM factory."""

from __future__ import annotations

import json

import numpy as np

from quam_builder.architecture.quantum_dots.components.xy_drive import XYDriveMW
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    TwoQubitMacroName,
    VoltagePointName,
)
from quam_builder.architecture.quantum_dots.operations.voltage_balanced_macros.two_qubit_macros import (
    BalancedCz2QMacro,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam

from .conftest import create_ld_quam


def test_create_ld_quam_serializes_physical_channels(tmp_path):
    """The wiring-built machine should contain the expected physical channels."""
    machine = create_ld_quam()
    state_path = tmp_path / "state_old.json"

    machine.save(state_path, include_defaults=False)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    physical_channels = state["physical_channels"]
    expected_channels = {
        "plunger_1",
        "plunger_2",
        "sensor_1",
        "barrier_1",
    }

    assert set(physical_channels) == expected_channels


def test_create_ld_quam_has_qubits_with_xy_drives(tmp_path):
    """Each qubit should have a MW-FEM-backed XY drive with the default reference pulse."""
    machine = create_ld_quam()

    for qname in ("q1", "q2"):
        qubit = machine.qubits[qname]
        assert qubit.xy is not None, f"{qname} should have an XY drive"
        assert isinstance(
            qubit.xy, XYDriveMW
        ), f"{qname} should use the MW-FEM XY drive"
        assert (
            DrivePulseName.GAUSSIAN in qubit.xy.operations
        ), f"{qname} should have the gaussian reference pulse"


def test_create_ld_quam_has_default_macros():
    """Each qubit should have the full set of default macros wired."""
    machine = create_ld_quam()

    required_macros = {
        VoltagePointName.INITIALIZE,
        VoltagePointName.MEASURE,
        VoltagePointName.EMPTY,
        SingleQubitMacroName.XY_DRIVE,
        SingleQubitMacroName.X_180,
        SingleQubitMacroName.X_90,
        SingleQubitMacroName.Y_180,
        SingleQubitMacroName.Y_90,
    }
    for qname in ("q1", "q2"):
        qubit = machine.qubits[qname]
        macro_names = set(qubit.macros.keys())
        missing = required_macros - macro_names
        assert not missing, f"{qname} missing macros: {missing}"


def test_create_ld_quam_registers_canonical_voltage_points():
    """Canonical voltage points should be materialized for qubits and dot pairs."""
    machine = create_ld_quam()

    shared_point_names = (
        VoltagePointName.INITIALIZE,
        VoltagePointName.MEASURE,
        VoltagePointName.EMPTY,
        VoltagePointName.EXCHANGE,
    )
    cz_only_gate_set = VoltagePointName.CZ

    for qubit in machine.qubits.values():
        gate_set_macros = qubit.voltage_sequence.gate_set.get_macros()
        for point_name in shared_point_names:
            assert qubit._create_point_name(point_name) in gate_set_macros

    for quantum_dot_pair in machine.quantum_dot_pairs.values():
        gate_set_macros = quantum_dot_pair.voltage_sequence.gate_set.get_macros()
        for point_name in shared_point_names:
            assert quantum_dot_pair._create_point_name(point_name) in gate_set_macros

    for qubit_pair in machine.qubit_pairs.values():
        gate_set_macros = qubit_pair.voltage_sequence.gate_set.get_macros()
        for point_name in (*shared_point_names, cz_only_gate_set):
            assert qubit_pair._create_point_name(point_name) in gate_set_macros


def test_create_ld_quam_has_default_qubit_pair_macros():
    """Each qubit pair should expose the latest enum-backed default macros."""
    machine = create_ld_quam()

    required_macros = {
        VoltagePointName.INITIALIZE,
        VoltagePointName.MEASURE,
        VoltagePointName.EMPTY,
        TwoQubitMacroName.EXCHANGE,
    }
    for qname in ("q1_q2",):
        qubit_pair = machine.qubit_pairs[qname]
        macro_names = set(qubit_pair.macros.keys())
        missing = required_macros - macro_names
        assert not missing, f"{qname} missing macros: {missing}"


# ── Wiring behaviour integration tests ────────────────────────────────


def test_ports_materialized_after_stage2():
    """All port references in machine.wiring should have corresponding port objects."""
    machine = create_ld_quam()

    def _collect_port_refs(obj, refs=None):
        if refs is None:
            refs = set()
        if isinstance(obj, dict):
            for v in obj.values():
                _collect_port_refs(v, refs)
        elif isinstance(obj, str) and obj.startswith("#/ports/"):
            refs.add(obj)
        return refs

    wiring_refs = _collect_port_refs(machine.wiring)
    for ref in wiring_refs:
        parts = ref.lstrip("#/").split("/")
        node = machine
        for part in parts:
            if isinstance(node, dict):
                assert part in node, f"Port reference {ref} not materialized"
                node = node[part]
            else:
                assert hasattr(node, part), f"Port reference {ref} not materialized"
                node = getattr(node, part)


def test_readout_resonators_not_sticky():
    """Readout resonators should not have sticky enabled."""
    machine = create_ld_quam()
    for sensor in machine.sensor_dots.values():
        rr = getattr(sensor, "readout_resonator", None)
        if rr is None:
            continue
        assert rr.sticky is None or not getattr(
            rr.sticky, "enabled", True
        ), f"Readout resonator {rr.id} should not have sticky enabled"


def test_sensor_dots_wired_to_quantum_dot_pairs():
    """Each qubit pair should have non-empty sensor_dots on its quantum_dot_pair."""
    machine = create_ld_quam()

    pair_q1_q2 = machine.qubit_pairs["q1_q2"]
    assert len(pair_q1_q2.quantum_dot_pair.sensor_dots) > 0
    assert "#/sensor_dots/virtual_sensor_1" in pair_q1_q2.quantum_dot_pair.sensor_dots


def test_preferred_readout_quantum_dot_set():
    """Each qubit in a pair should have preferred_readout_quantum_dot set."""
    machine = create_ld_quam()

    for qp in machine.qubit_pairs.values():
        qc = qp.qubit_control
        qt = qp.qubit_target
        assert (
            qc.preferred_readout_quantum_dot == qt.quantum_dot.id
        ), f"Control qubit {qc.id} preferred_readout_quantum_dot should be {qt.quantum_dot.id}"
        assert (
            qt.preferred_readout_quantum_dot == qc.quantum_dot.id
        ), f"Target qubit {qt.id} preferred_readout_quantum_dot should be {qc.quantum_dot.id}"


# ── Exchange decay model serialization round-trip ──────────────────────


_TEST_MODEL = {
    "type": "polynomial",
    "coeffs": [6130.86, -7109.04, 2647.79, -308.09],
    "degree": 3,
}

_TEST_VOLTAGES = [0.25, 0.30, 0.35, 0.40]


def _eval_poly(coeffs, v):
    """Horner evaluation matching BalancedCz2QMacro.t_2pi."""
    result = 0.0
    for c in coeffs:
        result = result * v + c
    return result


def test_exchange_decay_model_save_load_roundtrip(tmp_path):
    """exchange_decay_model should survive a QuAM save → load cycle.

    Verifies that the polynomial T_2π(V) model stored on the CZ macro
    is correctly serialized to JSON and deserialized back, and that
    t_2pi() produces identical results before and after reloading.
    """
    machine = create_ld_quam()
    pair = next(iter(machine.qubit_pairs.values()))

    # The base factory creates a generic CZMacro; swap to the
    # voltage-balanced variant that carries exchange_decay_model.
    cz_macro = BalancedCz2QMacro()
    pair.macros[TwoQubitMacroName.CZ.value] = cz_macro

    # 1. Store the model
    cz_macro.exchange_decay_model = dict(_TEST_MODEL)

    # 2. Evaluate t_2pi before saving
    t_before = [cz_macro.t_2pi(v) for v in _TEST_VOLTAGES]
    for v, t in zip(_TEST_VOLTAGES, t_before):
        assert np.isfinite(t), f"t_2pi({v}) should be finite before save"

    # 3. Save to disk
    save_dir = tmp_path / "quam_state"
    machine.save(save_dir, include_defaults=False)

    # 4. Verify the model appears in the serialized JSON
    state_json = json.loads((save_dir / "state_old.json").read_text(encoding="utf-8"))
    assert "exchange_decay_model" in json.dumps(state_json), (
        "exchange_decay_model not found in serialized state"
    )

    # 5. Load from disk
    loaded = LossDiVincenzoQuam.load(save_dir)
    loaded_pair = next(iter(loaded.qubit_pairs.values()))
    loaded_macro = loaded_pair.macros[TwoQubitMacroName.CZ.value]

    # 6. Verify the model was deserialized correctly
    loaded_model = loaded_macro.exchange_decay_model
    assert loaded_model is not None, "exchange_decay_model should not be None after load"
    assert loaded_model["type"] == _TEST_MODEL["type"]
    assert loaded_model["degree"] == _TEST_MODEL["degree"]
    assert len(loaded_model["coeffs"]) == len(_TEST_MODEL["coeffs"])
    for i, (got, expected) in enumerate(
        zip(loaded_model["coeffs"], _TEST_MODEL["coeffs"])
    ):
        assert abs(got - expected) < 1e-10, (
            f"coeffs[{i}] mismatch: {got} != {expected}"
        )

    # 7. Evaluate t_2pi after loading — must match pre-save values
    t_after = [loaded_macro.t_2pi(v) for v in _TEST_VOLTAGES]
    for v, tb, ta in zip(_TEST_VOLTAGES, t_before, t_after):
        assert abs(ta - tb) < 1e-10, (
            f"t_2pi({v}) changed after round-trip: {tb} → {ta}"
        )

    # 8. Cross-check against manual polynomial evaluation
    for v in _TEST_VOLTAGES:
        expected = _eval_poly(_TEST_MODEL["coeffs"], v)
        actual = loaded_macro.t_2pi(v)
        assert abs(actual - expected) < 1e-10, (
            f"t_2pi({v}) = {actual}, expected {expected}"
        )
