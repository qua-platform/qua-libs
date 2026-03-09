"""Regression tests for the programmatic simulation QuAM factory."""

from __future__ import annotations

import json

from .quam_factory import create_minimal_quam


def test_create_minimal_quam_serializes_qdac_specs(tmp_path):
    """Each physical DC gate should serialize a QDAC spec into state.json."""
    machine = create_minimal_quam()
    state_path = tmp_path / "state.json"

    machine.save(state_path, include_defaults=False)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    physical_channels = state["physical_channels"]
    expected_qdac_ports = {
        "plunger_1": 1,
        "plunger_2": 2,
        "sensor_DC_1": 3,
        "plunger_3": 4,
        "plunger_4": 5,
        "sensor_DC_2": 6,
    }

    assert set(physical_channels) == set(expected_qdac_ports)
    for gate_name, qdac_output_port in expected_qdac_ports.items():
        assert physical_channels[gate_name]["qdac_spec"] == {
            "qdac_output_port": qdac_output_port,
            "__class__": "quam_builder.architecture.quantum_dots.components.voltage_gate.QdacSpec",
        }
