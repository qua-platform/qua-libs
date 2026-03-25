"""Regression tests for the wiring-based QuAM factory."""

from __future__ import annotations

import json

from quam_builder.architecture.quantum_dots.components.xy_drive import XYDriveMW
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    TwoQubitMacroName,
    VoltagePointName,
)

from .conftest import create_ld_quam


def test_create_ld_quam_serializes_physical_channels(tmp_path):
    """The wiring-built machine should contain the expected physical channels."""
    machine = create_ld_quam()
    state_path = tmp_path / "state.json"

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
        assert isinstance(qubit.xy, XYDriveMW), f"{qname} should use the MW-FEM XY drive"
        assert DrivePulseName.GAUSSIAN in qubit.xy.operations, f"{qname} should have the gaussian reference pulse"


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

    for qubit in machine.qubits.values():
        gate_set_macros = qubit.voltage_sequence.gate_set.get_macros()
        for point_name in VoltagePointName:
            assert qubit._create_point_name(point_name) in gate_set_macros

    for quantum_dot_pair in machine.quantum_dot_pairs.values():
        gate_set_macros = quantum_dot_pair.voltage_sequence.gate_set.get_macros()
        for point_name in VoltagePointName:
            assert quantum_dot_pair._create_point_name(point_name) in gate_set_macros


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
