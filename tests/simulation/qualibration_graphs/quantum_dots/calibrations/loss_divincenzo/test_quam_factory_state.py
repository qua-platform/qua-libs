"""Regression tests for the wiring-based QuAM factory."""

from __future__ import annotations

import json

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
        "plunger_3",
        "plunger_4",
        "sensor_1",
        "sensor_2",
        "barrier_1",
        "barrier_2",
    }

    assert set(physical_channels) == expected_channels


def test_create_ld_quam_has_qubits_with_xy_drives(tmp_path):
    """Each qubit should have an XY drive with default pulses."""
    machine = create_ld_quam()

    for qname in ("q1", "q2", "q3", "q4"):
        qubit = machine.qubits[qname]
        assert qubit.xy is not None, f"{qname} should have an XY drive"
        assert "x180" in qubit.xy.operations, f"{qname} should have x180 pulse"
        assert "x90" in qubit.xy.operations, f"{qname} should have x90 pulse"


def test_create_ld_quam_has_default_macros():
    """Each qubit should have the full set of default macros wired."""
    machine = create_ld_quam()

    required_macros = {"initialize", "measure", "x180", "x90", "y180", "y90", "xy_drive"}
    for qname in ("q1", "q2", "q3", "q4"):
        qubit = machine.qubits[qname]
        macro_names = set(qubit.macros.keys())
        missing = required_macros - macro_names
        assert not missing, f"{qname} missing macros: {missing}"
